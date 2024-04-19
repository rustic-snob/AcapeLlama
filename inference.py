import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import argparse
import re

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from data.data_controller import DataCleaning, LyricsDataset
from torch.utils.data import DataLoader
from transformers import TextGenerationPipeline
from utils.evaluate import evaluate

import pandas as pd
import yaml
from tqdm.auto import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peft_model_path", type=str)
    parser.add_argument("--file_name", type=str, default='unpredicted')
    parser.add_argument("--eval", action='store_true')

    return parser.parse_args()

def tokenizing(x, CFG, eval=True):
        
        """ 
        Arguments:
        x: pd.DataFrame

        Returns:
        inputs: Dict({'input_ids', 'attention_mask', 'labels', ...}), 각 tensor(num_data, max_length)
        """
        rows = []
        if eval:
            for _, row in tqdm(x.iterrows()):
                rows.extend([{'title':row['title'],
                                'genre':row['genre'],
                                'lyrics_whole':row['lyrics'],
                                'mungchi':m,
                                'labels':l} for m, l in zip(row['line_sample_word_mungchi_integer'], row['line_sample_word_mungchi_string'])])
                rows.extend([{'title':row['title'],
                                'genre':row['genre'],
                                'lyrics_whole':row['lyrics'],
                                'mungchi':m,
                                'labels':l} for m, l in zip(row['line_sample_line_mungchi_integer'], row['line_sample_line_mungchi_string'])])
                rows.extend([{'title':row['title'],
                                'genre':row['genre'],
                                'lyrics_whole':row['lyrics'],
                                'mungchi':m,
                                'labels':l} for m, l in zip(row['verse_sample_word_mungchi_integer'], row['verse_sample_word_mungchi_string'])])
                rows.extend([{'title':row['title'],
                                'genre':row['genre'],
                                'lyrics_whole':row['lyrics'],
                                'mungchi':m,
                                'labels':l} for m, l in zip(row['verse_sample_line_mungchi_integer'], row['verse_sample_line_mungchi_string'])])
                if 'no_total' in CFG['data_config']['data_cleaning']:
                    continue
                rows.extend([{'title':row['title'],
                                'genre':row['genre'],
                                'lyrics_whole':row['lyrics'],
                                'mungchi':m,
                                'labels':l} for m, l in zip(row['total_sample_word_mungchi_integer'], row['total_sample_word_mungchi_string'])])
                rows.extend([{'title':row['title'],
                                'genre':row['genre'],
                                'lyrics_whole':row['lyrics'],
                                'mungchi':m,
                                'labels':l} for m, l in zip(row['total_sample_line_mungchi_integer'], row['total_sample_line_mungchi_string'])])
        else:
            for _, row in tqdm(x.iterrows()):
                rows.append({'title':row['title'],
                                'genre':row['genre'],
                                'mungchi':row['muncghi']})
        
        # Create a new DataFrame from the list of new rows
        x = pd.DataFrame(rows)
        
        # Reset the index of the new DataFrame
        x.reset_index(drop=True, inplace=True)
        
        if not CFG['train_config']['induce_align']:
            instruction = "다음 조건에 어울리는 가사를 써주실 수 있나요? 주어진 음절 수를 절대 벗어나면 안 돼요. 제목과 장르에 어울려야 해요. 꼭 [가사 / 가사 / 가사 / 가사]의 형태로 써주세요."
            
        else:
            instruction = "다음 조건에 어울리는 가사를 써주실 수 있나요? 주어진 음절 수를 절대 벗어나면 안 돼요. 제목과 장르에 어울려야 해요. 꼭 [(음절) 가사 / (음절) 가사 / (음절) 가사 / (음절) 가사]의 형태로 써주세요."
            
        prompt_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: {prompt}\nAssistant:\n"
        
        prompts_list = x.apply(lambda row: prompt_template.format(prompt=f"{instruction} 음절 수는 [{' / '.join(map(str, row['mungchi']))}], 제목은 [{row['title']}], 장르는 [{row['genre']}]에요."), axis=1)
            

        # if CFG['train_config']['induce_align']:
        #     x['labels'] = x.apply(lambda row: ' / '.join([f"({m}) {l}" for m, l in zip(row['mungchi'], row['label'].split(' / '))]), axis=1)
        
        return prompts_list.tolist(), x['title'].tolist(), x['genre'].tolist(), x['lyrics_whole'].tolist(), x['mungchi'].tolist(), x['labels'].apply(lambda x: x.split(' / ')).tolist()

def main():
    args = get_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    peft_model_id = args.peft_model_path
    peft_config = PeftConfig.from_pretrained(peft_model_id)

    print(f"Loading base model: {peft_config.base_model_name_or_path}")    
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device, non_blocking=True)

    print(f"Loading PEFT: {peft_model_id}")
    model = PeftModel.from_pretrained(base_model, peft_model_id)
    model = model.to(device)
    
    print(f"Running merge_and_unload ...")
    model = model.merge_and_unload()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, trust_remote_code=True)
    
    print(f"Loading samples ...")
    predict_df = pd.read_json(f"{(save_path:=args.peft_model_path.split('checkpoints')[0])}/{args.file_name}.json")
    
    with open(f"{save_path}/config.yaml") as f:
        CFG = yaml.load(f, Loqader=yaml.FullLoader)
        
    dc = DataCleaning(CFG['data_config']['data_cleaning'])
    predict_df[:10] = dc.process(predict_df) 
    prompts, titles, genres, lyrics_wholes, input_mungchi, gt_mungchi = tokenizing(predict_df, CFG)
    predict_dataset = LyricsDataset(prompts)
    predict_dataloader = DataLoader(predict_dataset, batch_size=16, shuffle=False)
    
    print(f"Generating ...")
    pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=device, max_new_tokens=200)

    preds = []
    for batch in tqdm(predict_dataloader):
        preds.extend([i[0]['generated_text'].split(' / ') if CFG['train_config']['induce_align'] else [re.sub('\(.*?\)\s*', '', j) for j in i[0]['generated_text'].split(' / ')] for i in pipe(batch, return_full_text=False)])
        
    if args.eval:
        print(f"Evaluating ...")
        raise NotImplementedError()
        evaluate(input_mungchi, preds, gt_mungchi)
    else:
        y = pd.DataFrame({'title':titles,'genre':genres, 'lyrics_whole':lyrics_wholes, 'input_mungchi':input_mungchi, 'gt_mungchi':gt_mungchi, 'pred_mungchi':preds})
        
    print(f"Jobs Done .")
        

if __name__ == "__main__" :
    main()