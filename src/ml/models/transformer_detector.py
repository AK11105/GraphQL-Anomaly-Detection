# ml/models/transformer_detector.py
import torch
import torch.nn as nn
from typing import Optional

class TransformerDetector(nn.Module):
    """
    Transformer encoder-based temporal detector.
    Input: (batch, seq_len, feature_dim)
    Output: (batch, 1) probability score
    """
    def __init__(
        self,
        feature_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Linear(feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # PyTorch 1.9+
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # will pool over time after transpose
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, feature_dim)
        x = self.embed(x)                 # -> (batch, seq_len, d_model)
        out = self.encoder(x)             # -> (batch, seq_len, d_model)
        # pool over time axis
        # transpose to (batch, d_model, seq_len) -> AdaptiveAvgPool1d -> (batch, d_model, 1)
        pooled = out.transpose(1, 2)
        pooled = self.pool(pooled).squeeze(-1)  # -> (batch, d_model)
        return self.fc(pooled)

    @staticmethod
    def from_config(cfg: dict) -> "TransformerDetector":
        return TransformerDetector(
            feature_dim=cfg["feature_dim"],
            d_model=cfg.get("d_model", 128),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 3),
            dim_feedforward=cfg.get("dim_feedforward", 256),
            dropout=cfg.get("dropout", 0.1),
        )
