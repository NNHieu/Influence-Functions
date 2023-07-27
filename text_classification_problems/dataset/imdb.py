from typing import Literal

import os.path
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd

MAP_LABELS = {
    'negative': 0,
    'positive':1
}

class IMDBDataset(Dataset):
    num_classes = len(MAP_LABELS)
    base_folder = "imdb"
    csv_name = {
        "train": "raw/train.csv",
        "denoised_train": "noise_label_1310.csv",
        "val": "raw/val.csv",
        "test": "raw/test.csv",
    }
    def __init__(self, root, split: Literal["train", "val", "test", "denoised_train"], tokenizer: AutoTokenizer, max_len=256):
        super().__init__()
        path = os.path.join(root, self.base_folder, self.csv_name[split])
        self.df = pd.read_csv(path)
        self.max_len = max_len
        self.tokenizer = tokenizer
        if "sentiment" in self.df.columns:
            self.df['label'] = self.df['sentiment'].apply(lambda x: MAP_LABELS[x])

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
        review_text = row["review"]
        encode = self.tokenizer(
            review_text,
            max_length = self.max_len,
            truncation = True,
            padding = 'max_length',
            add_special_tokens = True,
            return_token_type_ids = True,
            return_attention_mask = True,
            return_tensors = 'pt',

        )
        encode['input_ids'] = encode['input_ids'].squeeze()
        encode['attention_mask'] = encode['attention_mask'].squeeze()
        encode['token_type_ids'] = encode['token_type_ids'].squeeze()
        encode['label'] = int(row['label'])
        
        return encode
