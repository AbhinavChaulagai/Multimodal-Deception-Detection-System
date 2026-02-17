import torch
import torch.nn as nn
from transformers import BertModel

class TextModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, 256)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_token = outputs.pooler_output
        return self.fc(cls_token)