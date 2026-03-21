import torch
import torch.nn as nn
from typing import Optional

class FeatureExtractionLayer(nn.Module):
    """
    One feature-extraction block:
      x -> (MHA + residual) -> (MLP + residual)
    Input:  (B, D)
    Output: (B, D)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if mlp_hidden_dim is None:
            mlp_hidden_dim = dim

        self.use_layernorm = use_layernorm
        self.ln1 = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(dim) if use_layernorm else nn.Identity()

        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True
        )
        self.attn_drop = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        if x.ndim != 2:
            raise ValueError(f"Expected x shape (B,D), got {tuple(x.shape)}")

        # ----- MHA sub-layer (Pre-LN) -----
        h = self.ln1(x)
        h = h.unsqueeze(1)  # (B, 1, D)
        attn_out, _ = self.mha(h, h, h, need_weights=False)  # (B, 1, D)
        attn_out = attn_out.squeeze(1)  # (B, D)
        x = x + self.attn_drop(attn_out)  # residual

        # ----- MLP sub-layer (Pre-LN) -----
        h = self.ln2(x)
        x = x + self.mlp(h)  # residual
        return x


class FeatureExtractionModule(nn.Module):
    """
    Stacked feature-extraction blocks.
    Input:  (B, D)
    Output: (B, D)
    """
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        num_layers: int = 1,
        mlp_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.layers = nn.ModuleList([
            FeatureExtractionLayer(
                dim=input_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
                use_layernorm=use_layernorm,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
