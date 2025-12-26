import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    One residual block for tabular ResNet-MLP:
    x -> LN -> Linear -> GELU -> Dropout -> Linear -> Dropout -> +x
    """
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        h = self.norm(x)
        h = self.fc1(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return residual + h


class FeatureResMLP(nn.Module):
    """
    Strong residual MLP for tabular GraphQL feature vectors.

    Input:
        x: (batch_size, input_dim)
    Output:
        logit: (batch_size,)   # use BCEWithLogitsLoss
    """

    def __init__(
        self,
        input_dim: int,
        width: int = 128,
        num_blocks: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, width)

        # Residual trunk
        self.blocks = nn.ModuleList([
            ResidualBlock(width, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # Output head
        self.norm_out = nn.LayerNorm(width)
        self.head = nn.Sequential(
            nn.Linear(width, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),   # binary logit
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: FloatTensor of shape (batch_size, input_dim)
        returns: logits (batch_size,)
        """
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        logit = self.head(h).squeeze(-1)
        return logit
