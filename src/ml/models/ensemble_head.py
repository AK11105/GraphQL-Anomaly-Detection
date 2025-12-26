import torch
import torch.nn as nn
import torch.nn.functional as F


class StrongEnsembleHead(nn.Module):
    """
    Stacking-based ensemble head.

    Inputs (all are probabilities in [0,1]):
        p_feature      -> from FeatureResMLP
        p_lstm         -> from AttentiveBiLSTM
        p_transformer  -> from SOTA Transformer

    Output:
        logit -> final malicious probability logit
    """

    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, p_feature=None, p_lstm=None, p_transformer=None):
        """
        Each input is expected to be:
            Tensor of shape (batch,) with values in [0,1]

        Any of the inputs may be None (for ablation or partial deployment).

        Returns:
            logit: (batch,)
        """
        parts = []

        if p_feature is not None:
            parts.append(p_feature.unsqueeze(-1))

        if p_lstm is not None:
            parts.append(p_lstm.unsqueeze(-1))

        if p_transformer is not None:
            parts.append(p_transformer.unsqueeze(-1))

        if len(parts) == 0:
            raise ValueError("At least one probability input must be provided.")

        x = torch.cat(parts, dim=-1)  # (batch, k)
        logit = self.net(x).squeeze(-1)
        return logit
