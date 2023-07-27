import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
import numpy as np
import os
from typing import Literal


def get_labels():
    return [
        "O",
        "B-MISC",
        "I-MISC",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
    ]

def load_data_from_file(file_name):
    f = open(file_name, "r")
    examples = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith("-DOCSTART") or line[0] == "\n":
            # if the end of sentence, we append this sentence to examples and reset all of lists
            if len(sentence) > 0:
                examples.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.split(" ")
        sentence.append(splits[0])
        # Using NER label
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        examples.append((sentence, label))
        sentence = []
        label = []

    return examples


class CoNLL2003(Dataset):
    num_labels = len(get_labels())
    base_folder = "conll2003"
    file_name = {
        "train": "raw/train.txt",
        "val": "raw/valid.txt",
        "test": "raw/test.txt",
    }

    def __init__(
        self,
        root,
        split: Literal["train", "val", "test"],
        tokenizer: BertTokenizerFast,
        max_len=256,
    ):
        super().__init__()
        path = os.path.join(root, self.base_folder, self.file_name[split])
        self.examples = load_data_from_file(path)
        self.max_len = max_len
        # self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.tokenizer = tokenizer
        # Build dictionary of labels
        self.labels_to_ids = {label: i for i, label in enumerate(get_labels())}
        self.ids_to_labels = {i: label for i, label in enumerate(get_labels())}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        # step 1: get the sentence and word labels
        sentence = self.examples[index][0]
        word_labels = self.examples[index][1]
        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len
                                  )
        # step 3: create token labels only for first word pieces of each tokenized word
        labels = [self.labels_to_ids[label] for label in word_labels]
        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        # create an empty array of -100 of length max_length
        # Word pieces that should be ignored have a label of -100 (which is the default ignore_index of PyTorch's CrossEntropyLoss).
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        if self.read_flipped_features:
            encoded_flipped = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100

        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                # overwrite label
                encoded_labels[idx] = labels[i]
                i += 1

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels) #(bs, 128)

        return item

