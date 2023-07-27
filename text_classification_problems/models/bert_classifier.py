import torch.nn as nn
from transformers import BertConfig, BertModel
import torch


class BertClassifier(nn.Module):
    def __init__(self, num_classes, path_pretrained_or_name, name):
        super().__init__()
        config = BertConfig.from_pretrained(path_pretrained_or_name)
        self.bert = BertModel.from_pretrained(
            path_pretrained_or_name,
            config=config
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(config.hidden_size, num_classes)
        nn.init.normal_(self.fc.weight, std=0.2)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask, token_type_ids)
        x = self.dropout(out.pooler_output)
        out = self.fc(x)
        return out

    def feature(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask, token_type_ids)
        return out.pooler_output
    
    def feature2logits(self, features):
        return self.fc(self.dropout(features))