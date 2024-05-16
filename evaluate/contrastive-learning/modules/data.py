import pandas as pd
from torch.utils.data import DataLoader, Dataset, Sampler
import re
import random
from tqdm import tqdm
import torch

# combine dataset
def get_combined_train_dataset(subset_line, subset_verse, subset_total, is_save=True, dir_path='/workspace/codes/data/'):
    
    dataframe_line = pd.DataFrame(subset_line['train'])
    dataframe_verse = pd.DataFrame(subset_verse['train'])
    dataframe_total = pd.DataFrame(subset_total['train'])

    combined_dataframe = pd.concat([dataframe_line, dataframe_verse, dataframe_total], ignore_index=True)

    if is_save:
        combined_dataframe.to_csv(f'{dir_path}/combined_train_dataframe.csv', index=False)
        
    return combined_dataframe

# combine dataset
def get_combined_test_dataset(subset_line, subset_verse, subset_total, is_save=True, dir_path='/workspace/codes/data/'):
    
    dataframe_line = pd.DataFrame(subset_line['test'])
    dataframe_verse = pd.DataFrame(subset_verse['test'])
    dataframe_total = pd.DataFrame(subset_total['test'])

    combined_dataframe = pd.concat([dataframe_line, dataframe_verse, dataframe_total], ignore_index=True)

    if is_save:
        combined_dataframe.to_csv(f'{dir_path}/combined_test_dataframe.csv', index=False)
        
    return combined_dataframe

# pre-processing
def preprocess(combined_dataframe):
    only_output_lyrics_combined_dataframe = combined_dataframe.drop(columns=['title', 'genre', 'mungchi', 'instruction','__index_level_0__'])

    only_output_lyrics_combined_dataframe['output'] = only_output_lyrics_combined_dataframe['output'].apply(lambda x: re.sub(r'\(\d+\)', '', x).replace(' / ', '').strip())

    only_output_lyrics_combined_dataframe['lyrics'] = only_output_lyrics_combined_dataframe['lyrics'].apply(lambda x: x.replace('\n{2,}', '\n').strip())
    only_output_lyrics_combined_dataframe['lyrics'] = only_output_lyrics_combined_dataframe['lyrics'].apply(lambda x: x.replace('\n', ' ').strip())
    only_output_lyrics_combined_dataframe['lyrics'] = only_output_lyrics_combined_dataframe['lyrics'].apply(lambda x: x.replace('"', '').strip())
    
    return only_output_lyrics_combined_dataframe

class LyricsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, token_bsz=100, tokenized_dataset_path=None):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.token_bsz = token_bsz

        if tokenized_dataset_path and self._check_tokenized_data(tokenized_dataset_path):
            self.outputs, self.lyrics = torch.load(tokenized_dataset_path)
        else:
            self.outputs = self.tokenize_column(dataframe['output'], token_bsz=self.token_bsz)
            self.lyrics = self.tokenize_column(dataframe['lyrics'], token_bsz=self.token_bsz)
            if tokenized_dataset_path:
                self._save_tokenized_data(tokenized_dataset_path, self.outputs, self.lyrics)

    def tokenize_column(self, column, token_bsz):
        all_tokens = []
        for i in tqdm(range(0, len(column), token_bsz), desc="Tokenizing"):
            batch = column[i:i + token_bsz].tolist()
            tokens = self.tokenizer(batch, add_special_tokens=True, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
            all_tokens.append(tokens)
        # Concatenate all batches into a single tensor
        concatenated_tokens = {key: torch.cat([batch[key] for batch in all_tokens], dim=0) for key in all_tokens[0].keys()}
        return concatenated_tokens

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        output = {key: val[idx].clone() for key, val in self.outputs.items()}
        label = {key: val[idx].clone() for key, val in self.lyrics.items()}
        return output, label

    def _save_tokenized_data(self, path, outputs, lyrics):
        torch.save((outputs, lyrics), path)

    def _check_tokenized_data(self, path):
        try:
            torch.load(path)
            return True
        except:
            return False

class LyricsBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices_by_label = {label: [] for label in set(dataset.dataframe['lyrics'].astype('category').cat.codes.values)}
        self.lyrics_labels = dataset.dataframe['lyrics'].astype('category').cat.codes.values  # 추가된 부분
        for idx, label in enumerate(self.lyrics_labels):
            self.indices_by_label[label].append(idx)
        
        self.used_indices = set()
        self.all_indices = set(range(len(dataset)))

        self.number_of_wasted_random_choices = 0

    def __iter__(self):
        for indices in self.indices_by_label.values():
            random.shuffle(indices)
        while self.all_indices:
            batch = []
            labels_used = set()
            for label, indices in self.indices_by_label.items():
                if labels_used.intersection(set([label])):
                    self.number_of_wasted_random_choices += 1
                    continue
                for idx in indices:
                    if idx in self.used_indices:
                        self.number_of_wasted_random_choices += 1
                        continue
                    if idx in self.all_indices:
                        batch.append(idx)
                        self.used_indices.add(idx)
                        self.all_indices.remove(idx)
                        labels_used.add(label)
                        break
                if len(batch) >= self.batch_size:
                    break

            if batch:
                yield batch
                if len(batch) < self.batch_size:
                    break

    def __len__(self):
        return len(self.dataset) // self.batch_size
    
    def get_usage_percent(self):
        return len(self.used_indices) / len(self.flat_indices)
    
    def get_number_of_wasted_random_choices(self):
        return self.number_of_wasted_random_choices

def set_loader(dataframe, batch_size, num_workers, tokenizer, token_bsz, tokenized_dataset_path):
    dataset = LyricsDataset(dataframe, tokenizer, token_bsz, tokenized_dataset_path)
    batch_sampler = LyricsBatchSampler(dataset, batch_size)
    
    data_loader = DataLoader(
        dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=True
    )
    
    return data_loader