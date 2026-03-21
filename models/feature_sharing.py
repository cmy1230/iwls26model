import torch
import torch.nn as nn
from typing import Optional


class _MLP(nn.Module):
    """Simple 2-layer MLP with residual-friendly output dim."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureSharingLayer(nn.Module):
    """
    Strictly follows paper equations (6)(7)(8):

    For each target task i:
      (6) A_k = MHA(F_k, F_k, F_i) for k=1..n
      (7) Ahat_i = MLP1(Concat(A_1..A_n)) + A_i
      (8) F_i_next = MLP2(Ahat_i) + F_i

    All dims are kept the same: F_i in R^d, F_i_next in R^d
    Input:  (B, n, d)
    Output: (B, n, d)
    """
    def __init__(
        self,
        dim: int,
        num_tasks: int,
        num_heads: int,
        mlp_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_tasks = num_tasks

        # Use one MHA module; we will call it n*n times with different (Q,K,V).
        # batch_first=True => inputs are (B, L, d)
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        # (7) MLP over concatenated {A_k} => input dim = n * d, output dim = d
        self.mlp_fuse = _MLP(in_dim=num_tasks * dim, out_dim=dim, hidden_dim=mlp_hidden_dim, dropout=dropout)

        # (8) MLP over Ahat_i => input dim = d, output dim = d
        self.mlp_update = _MLP(in_dim=dim, out_dim=dim, hidden_dim=mlp_hidden_dim, dropout=dropout)

    def forward(self, task_features: torch.Tensor) -> torch.Tensor:
        """
        task_features: (B, n, d)
        returns:       (B, n, d)
        """
        if task_features.ndim != 3:
            raise ValueError(f"task_features must be 3D (B,n,d), got shape={tuple(task_features.shape)}")
        B, n, d = task_features.shape
        if n != self.num_tasks or d != self.dim:
            raise ValueError(
                f"Expected (B,{self.num_tasks},{self.dim}), got (B,{n},{d}). "
                f"Please construct layer with matching num_tasks/dim."
            )

        # We will compute F_next for all i in [0..n-1]
        out_all = []

        # Pre-slice all F_k as (B,1,d) tokens for MHA calls
        F_tokens = task_features.unsqueeze(2)  # (B, n, 1, d)
        # So: F_tokens[:, k] is (B,1,d)

        for i in range(n):
            Fi = F_tokens[:, i]  # (B,1,d)

            A_list = []
            Ai = None

            # (6) compute A_k for k=0..n-1
            for k in range(n):
                Fk = F_tokens[:, k]  # (B,1,d)

                # A_k = MHA(Q=F_k, K=F_k, V=F_i)
                # Shapes: Q=(B,1,d), K=(B,1,d), V=(B,1,d) -> out=(B,1,d)
                Ak, _ = self.mha(query=Fk, key=Fk, value=Fi, need_weights=False)
                Ak = Ak.squeeze(1)  # (B,d)

                if k == i:
                    Ai = Ak
                A_list.append(Ak)

            if Ai is None:
                raise RuntimeError("Internal error: Ai not set")

            # (7) fuse: Ahat_i = MLP1(Concat(A_1..A_n)) + A_i
            concatA = torch.cat(A_list, dim=-1)          # (B, n*d)
            Ahat_i = self.mlp_fuse(concatA) + Ai         # (B, d)

            # (8) update: F_i_next = MLP2(Ahat_i) + F_i
            Fi_vec = task_features[:, i, :]              # (B, d)
            Fi_next = self.mlp_update(Ahat_i) + Fi_vec   # (B, d)

            out_all.append(Fi_next)

        # Stack back to (B, n, d)
        return torch.stack(out_all, dim=1)



class FeatureSharingModule(nn.Module):
    """
    Stacks multiple FeatureSharingLayer (depth = num_layers).
    Input/Output: (B, n, d)
    """
    def __init__(
        self,
        dim: int,
        num_tasks: int,
        num_heads: int,
        num_layers: int,
        mlp_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.layers = nn.ModuleList([
            FeatureSharingLayer(
                dim=dim,
                num_tasks=num_tasks,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, task_features: torch.Tensor) -> torch.Tensor:
        x = task_features
        for layer in self.layers:
            x = layer(x)
        return x