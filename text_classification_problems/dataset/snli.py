from typing import Literal

import os.path
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer

MAP_LABELS = {
    'entailment': 0,
    'neutral': 1,
    'contradiction': 2,
}

class SNLIDataset(Dataset):
    num_classes = len(MAP_LABELS)
    base_folder = "snli"
    csv_name = {
        "train": "raw/train.csv",
        "val": "raw/dev.csv",
        "test": "raw/test.csv",
    }
    def __init__(self, root, split: Literal["train", "val", "test"], tokenizer: AutoTokenizer, max_len=256):
        super().__init__()
        path = os.path.join(root, self.base_folder, self.csv_name[split])
        self.df = pd.read_csv(path)
        # convert label from text to int32
        self.df['label'] = self.df['gold_label'].apply(lambda x: MAP_LABELS[x])
        self.max_len = max_len
        # self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Creare sample input for a row in dataset
        [CLS] Sentence 1 [SEP] Sentence2 [SEP] [PAD] ...
        
        --> Example:
        Input tokens: [ ‘[CLS]’,  ‘Man’,  ‘is’,  ‘wearing’,  ‘blue’,  ‘jeans’,  ‘.’,  ‘[SEP]’,  ‘Man’,  ‘is’,  ‘wearing’,  ‘red’,  ‘jeans’, ‘.’,   ‘[SEP]’ ]
        Attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        Token type ids: [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        Keys output: input_ids, attention_mask, token_type_ids, [label]
        """
        row = self.df.iloc[index]
        text1 = row['sentence1']
        text2 = row['sentence2']
        data = self.tokenizer(
                text1, text2,
                max_length = self.max_len,
                truncation = True,
                padding = 'max_length',
                add_special_tokens = True,
                return_token_type_ids = True,
                return_attention_mask = True,
                return_tensors = 'pt'
            )
        data['input_ids'] = data['input_ids'][0]
        data['attention_mask'] = data['attention_mask'][0]
        data['token_type_ids'] = data['token_type_ids'][0]
        data['label'] = int(row['label'])
        return data