# src/ml/models/schema_text_classifier.py

import torch.nn as nn
from transformers import AutoModel


class SchemaTextClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.head = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled = out.last_hidden_state[:, 0]  # CLS
        return self.head(pooled)
