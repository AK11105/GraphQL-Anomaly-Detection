import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class SOTATransformerClassifier(nn.Module):
    """
    Pretrained Transformer-based binary classifier.

    - Uses a strong encoder (e.g. roberta-base or deberta-v3-base)
    - Adds a small MLP head on top of the pooled representation
    - Outputs logits for BCEWithLogitsLoss

    Forward:
        input_ids: LongTensor (batch, seq_len)
        attention_mask: LongTensor (batch, seq_len) with 1 for tokens, 0 for padding

    Returns:
        logit: FloatTensor (batch,)
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        dropout: float = 0.2,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        # Load config and backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)

        hidden_size = self.config.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        # Optionally freeze encoder (e.g. for ablations / low-resource)
        if freeze_encoder:
            for param in self.transformer.parameters():
                param.requires_grad = False

        self._init_head()

    def _init_head(self):
        # Initialize only the head; encoder stays as loaded
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask):
        """
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len)

        Returns:
            logits: (batch,)
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # For many models:
        # - If pooler_output exists -> use it
        # - Else -> use [CLS] token representation
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output  # (batch, hidden)
        else:
            # last_hidden_state: (batch, seq_len, hidden)
            pooled = outputs.last_hidden_state[:, 0, :]  # CLS

        logits = self.classifier(pooled).squeeze(-1)
        return logits
