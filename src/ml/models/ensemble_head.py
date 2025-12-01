# ml/models/ensemble_head.py
import torch
import torch.nn as nn

class EnsembleHead(nn.Module):
    """
    Combine static features (per-window or aggregated) and ML temporal score.

    static_vec: (batch, static_dim)
    ml_score:   (batch, 1)   - output of LSTM/Transformer

    Output: final score (batch, 1)
    """
    def __init__(self, static_dim: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(static_dim + 1, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2 if hidden >= 4 else 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2 if hidden >= 4 else 1, 1),
            nn.Sigmoid()
        )

        # init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, static_vec: torch.Tensor, ml_score: torch.Tensor) -> torch.Tensor:
        if ml_score.ndim == 1:
            ml_score = ml_score.unsqueeze(1)
        x = torch.cat([static_vec, ml_score], dim=1)
        return self.net(x)
