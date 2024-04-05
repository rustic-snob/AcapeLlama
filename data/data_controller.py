import re
import pickle
import torch
import yaml
import pandas as pd
import pytorch_lightning as pl
import re

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

class LyricsDataset(Dataset):
    """
    Dataloader에서 불러온 데이터를 Dataset으로 만들기
    """

    def __init__(self, inputs, train=False):
        self.inputs = inputs
        self.train = train

    def __getitem__(self, idx):
        inputs = {key: val[idx].clone().detach()
                for key, val in self.inputs.items()}
        
        return inputs
        
    def __len__(self):
        return len(self.inputs['input_ids'])

class Dataloader(pl.LightningDataModule):
    
    """
    원본 데이터를 불러와 전처리 후 Dataloader 만들어 LyricsDataset에 넘겨 최종적으로 사용할 데이터셋 만들기
    """

    def __init__(self, tokenizer, CFG):
        super(Dataloader, self).__init__()
        self.CFG = CFG
        self.tokenizer = tokenizer
        
        self.train_valid_df, self.predict_df = load_data(CFG)

    def tokenizing(self, x, train=False):
        
        """ 
        Arguments:
        x: pd.DataFrame

        Returns:
        inputs: Dict({'input_ids', 'attention_mask', 'labels', ...}), 각 tensor(num_data, max_length)
        """
        
        rows = []
        for _, row in x.iterrows():
            for m, l in zip(row['mungchi'], row['label']):
                new_row = row.to_dict()  
                new_row['mungchi'] = list(map(str,m)) 
                new_row['label'] = l     
                rows.append(new_row) 

        # Create a new DataFrame from the list of new rows
        x = pd.DataFrame(rows)

        # Reset the index of the new DataFrame
        x.reset_index(drop=True, inplace=True)
        
        if not self.CFG['train_config']['induce_align']:
            instruction = "다음 조건에 어울리는 가사를 써주실 수 있나요? 주어진 음절 수를 절대 벗어나면 안 돼요. 제목과 장르에 어울려야 해요. 꼭 [가사 / 가사 / 가사 / 가사]의 형태로 써주세요."
            
        else:
            instruction = "다음 조건에 어울리는 가사를 써주실 수 있나요? 주어진 음절 수를 절대 벗어나면 안 돼요. 제목과 장르에 어울려야 해요. 꼭 [(음절) 가사 / (음절) 가사 / (음절) 가사 / (음절) 가사]의 형태로 써주세요."
            
        prompt_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: {prompt}\nAssistant:\n"
        
        prompts_list = x.apply(lambda row: prompt_template.format(prompt=f"{instruction} 음절 수는 [{' / '.join(row['mungchi'])}], 제목은 [{row['title']}], 장르는 [{row['genre']}]에요."), axis=1)
            
        if train:
            if self.CFG['induce_align']:
                x['labels'] = x.apply(lambda row: ' / '.join([f"({m}) {l}" for m, l in zip(row['mungchi'], row['label'])]), axis=1)
            else:
                x['labels'] = x.apply(lambda row: ' / '.join(row['label']), axis=1)
                
            answers_list = x['labels'].tolist()
            
            inputs = self.tokenizer(
                [p+a for p, a in zip(prompts_list, answers_list)],
                return_tensors='pt',
                padding=True,
                max_length="longest",
                add_special_tokens=True,
            )
            inputs["labels"] = inputs["input_ids"].detach().clone()

            prompts_len = [len(self.tokenizer.encode(p, add_special_tokens=True)) for p in prompts_list]

            for i, prompt_len in enumerate(prompts_len):
                inputs["labels"][i, :prompt_len] = -100  

            return inputs

        else:          
            inputs = self.tokenizer(
                prompts_list.tolist(),
                return_tensors='pt',
                padding=True,
                max_length="longest",
                add_special_tokens=True,
            )

            return inputs

    def preprocessing(self, x, train=False):
        dc = DataCleaning(self.CFG['data_config']['data_cleaning'])

        if train:
            x = dc.process(x)    
            train_x, val_x = train_test_split(x,
                                            test_size=self.CFG['train']['test_size'],
                                            shuffle=True,
                                            random_state=self.CFG['seed'])
            
            train_inputs = self.tokenizing(train_x, train=True)
            val_inputs = self.tokenizing(val_x, train=True)

            return train_inputs, val_inputs
        
        else:
            x = dc.process(x) 
            predict_inputs = self.tokenizing(x, train=False)
            
            return predict_inputs

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터 준비
            train, val = self.preprocessing(self.train_valid_df, train=True)
            self.train_dataset = LyricsDataset(train, train=True)
            self.val_dataset = LyricsDataset(val, train=True)
        else:
            # 평가 데이터 호출
            predict = self.preprocessing(self.predict_df)
            self.predict_dataset = LyricsDataset(predict)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.CFG['train']['batch_size'], shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.CFG['train']['batch_size'], shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.CFG['train']['batch_size'], shuffle=False)
    
class DataCleaning():
    def __init__(self, select_list):
        self.select_list = select_list
        
    def process(self, df):
        for method_name in self.select_list:

            method = getattr(self, method_name, None)
            if method:
                df = method(df)
            else:
                print(f"Method {method_name} not found.")
        return df
    """
    data cleaning 코드
    """
    def genre_filtering(self, df):
        df = df[~df['genre'].str.contains('랩')]
        df = df[~df['genre'].str.contains('CCM')]
        df = df[~df['genre'].str.contains('J-POP')]
        df = df[~df['genre'].str.contains('-')]
                    
        return df


def load_data(CFG):
    """
    학습 데이터와 테스트 데이터 DataFrame 가져오기
    """
        
    df = pd.read_json(f"./data/ready/{CFG['data_config']['dataset_name']}.json", encoding='utf-8-sig') 
    df.dropna(inplace=True)
    if CFG['test_size']:
        train_valid_df, predict_df = train_test_split(df, shuffle = True, test_size=CFG['train_config']['test_size'], random_state=CFG['seed'])
        
        return train_valid_df, predict_df
    return df, None