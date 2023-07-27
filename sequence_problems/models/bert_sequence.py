import torch.nn as nn
import torch
# from transformers import BertForTokenClassification
from models.sequence_modeling_bert import BertForTokenClassification

class BertSequence(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.bert = BertForTokenClassification.from_pretrained(
            'bert-base-uncased',
            num_labels = num_labels,
            return_dict=False
        )

    def forward(self, input_ids, attention_mask, labels, reduction_loss=None): 
        # assign reduction_loss = mean to get loss of each token
        loss, tr_logits = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            reduction_loss=reduction_loss
        )
        return loss, tr_logits
    
    def predict(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs
