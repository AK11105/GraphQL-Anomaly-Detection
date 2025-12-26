import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_resmlp import ResidualBlock


class FeatureSchemaClassifier(nn.Module):
    """
    Residual MLP for schema identification (multiclass).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        width: int = 128,
        num_blocks: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, width)

        self.blocks = nn.ModuleList([
            ResidualBlock(width, dropout)
            for _ in range(num_blocks)
        ])

        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.head(h)
