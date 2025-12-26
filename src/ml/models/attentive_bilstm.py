import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentiveBiLSTM(nn.Module):
    """
    Strong text classifier:
    Embedding -> BiLSTM -> Multi-Head Self-Attention -> Masked Mean Pool -> MLP Head

    Input:
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len) with 1 for tokens, 0 for padding

    Output:
        logit: (batch,)  # binary logit for BCEWithLogitsLoss
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_size * 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: LongTensor (batch, seq_len)
        attention_mask: LongTensor (batch, seq_len), 1 = keep, 0 = pad
        """

        # (batch, seq_len, embed_dim)
        x = self.embedding(input_ids)

        # (batch, seq_len, 2 * hidden)
        lstm_out, _ = self.lstm(x)

        # Convert to key padding mask: True = ignore
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = None

        # Self-attention
        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=key_padding_mask
        )

        # Masked mean pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            attn_out = attn_out * mask
            pooled = attn_out.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            pooled = attn_out.mean(dim=1)

        pooled = self.norm(pooled)

        # Final logit
        logit = self.classifier(pooled).squeeze(-1)
        return logit
