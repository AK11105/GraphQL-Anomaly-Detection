# ml/models/lstm_detector.py
import torch
import torch.nn as nn
from typing import Optional

class LSTMDetector(nn.Module):
    """
    LSTM-based detector.

    Input:
        x: float tensor shape (batch, seq_len, feature_dim)

    Output:
        score: float tensor shape (batch, 1) with sigmoid in [0,1]
    """
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * self.num_directions, 1)
        self.sigmoid = nn.Sigmoid()

        # weight init
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0.0)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, feature_dim)
        out, _ = self.lstm(x)            # out: (batch, seq_len, hidden_dim * num_directions)
        last = out[:, -1, :]             # take last time-step
        last = self.dropout(last)
        score = self.sigmoid(self.fc(last))
        return score

    @staticmethod
    def from_config(cfg: dict) -> "LSTMDetector":
        return LSTMDetector(
            feature_dim=cfg["feature_dim"],
            hidden_dim=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.3),
            bidirectional=cfg.get("bidirectional", False),
        )
