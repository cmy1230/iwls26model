from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import os

import torch
import torch.nn as nn
import dgl

from models.gin import GIN
from models.lstm import LSTMModel
from models.feature_extraction import FeatureExtractionModule
from models.feature_sharing import FeatureSharingModule
from models.ensemble_prediction import EnsemblePredictionModule

# helpers
def graph_readout_mean(g: dgl.DGLGraph, node_feat: torch.Tensor) -> torch.Tensor:
    """
    Compute a graph-level embedding from node embeddings.
    Default: mean pooling.

    Args:
      g: batched DGLGraph
      node_feat: (N_total, D)

    Returns:
      graph_feat: (B, D)
    """
    with g.local_scope():
        g.ndata["h"] = node_feat
        hg = dgl.mean_nodes(g, "h")
    return hg



class TopCircuitSeqModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.T = cfg.num_tasks

        # -------------------------
        # Encoders
        # -------------------------
        self.gin = GIN(
            input_dim=cfg.gin_in_dim,
            output_dim=cfg.gin_hidden_dim,
            num_layers=cfg.gin_layers,
        )
        self.lstm = LSTMModel(
            input_dim=cfg.seq_in_dim,
            output_dim=cfg.seq_hidden_dim,
            num_layers=cfg.seq_layers,
        )

        # -------------------------
        # direct concat, no projection
        # -------------------------
        self.fusion_in = cfg.gin_hidden_dim + cfg.seq_hidden_dim
        self.D = self.fusion_in
        self.fuse = nn.Identity()

        # -------------------------
        # Per-task Feature Extraction
        # -------------------------
        self.feature_extractors = nn.ModuleList([
            FeatureExtractionModule(input_dim=self.D, num_heads=cfg.fe_num_heads, num_layers=cfg.fe_num_layers)
            for _ in range(self.T)
        ])

        # -------------------------
        # Feature Sharing across tasks
        # Input/Output: (B, T, D)
        # -------------------------
        self.feature_sharing = FeatureSharingModule(
            dim=self.D,
            num_tasks=self.T,
            num_heads=cfg.fs_num_heads,
            num_layers=cfg.fs_num_layers,
            mlp_hidden_dim=cfg.fs_mlp_hidden_dim,
            dropout=cfg.fs_dropout,
            attn_dropout=cfg.fs_attn_dropout,
        )

        # -------------------------
        # Per-task prediction heads (ensemble)
        # -------------------------
        self.ensembles = nn.ModuleList([
            EnsemblePredictionModule(
                input_dim=self.D,
                output_dim=1,
                num_classes=cfg.ens_num_classes,
                num_layers=cfg.ens_num_layers,
                hidden_dim=cfg.ens_hidden_dim,
            )
            for _ in range(self.T)
        ])

    def _graph_pool(self, g: dgl.DGLGraph, node_emb: torch.Tensor) -> torch.Tensor:
        if self.cfg.graph_pool == "mean":
            return graph_readout_mean(g, node_emb)
        raise ValueError(f"Unknown graph_pool={self.cfg.graph_pool}")

    def forward(
        self,
        g: Optional[dgl.DGLGraph],
        seq: torch.Tensor,
        seq_len: Optional[torch.Tensor] = None,
        target_bins: Optional[torch.Tensor] = None,
        g_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            g: batched DGLGraph；若提供 ``g_emb`` 则可为 None（跳过 GIN，用于同电路预计算缓存）。
            seq: (B, L, seq_in_dim) 序列特征
            seq_len: (B,) 序列长度（可选）
            target_bins: (B, T) 每个样本每个任务的分桶索引，训练时提供用于 teacher forcing。
                        如果为 None，则使用 classifier 的 argmax 选择分支（推理模式）。
            g_emb: 可选，形状 (B, gin_hidden_dim)。与 ``g`` 二选一：提供时直接使用预计算的图级向量。

        Returns:
            values: (B, T, out_dim) 回归输出
            logits: (B, T, C) 分类 logits，如果 num_classes == 1 则为 None
        """
        # --------- Graph encoder ---------
        if g_emb is not None:
            g_emb = g_emb.to(device=seq.device, dtype=torch.float32)
            if g_emb.ndim != 2 or g_emb.shape[0] != seq.shape[0]:
                raise ValueError(
                    f"g_emb must be (B, gin_hidden_dim), got {tuple(g_emb.shape)} for B={seq.shape[0]}"
                )
            if g_emb.shape[1] != self.cfg.gin_hidden_dim:
                raise ValueError(
                    f"g_emb last dim must be gin_hidden_dim={self.cfg.gin_hidden_dim}, got {g_emb.shape[1]}"
                )
        else:
            if g is None:
                raise ValueError("forward requires either batched graph g or tensor g_emb")
            if "nf" not in g.ndata:
                raise KeyError("Graph missing node feature 'nf' in g.ndata. Please set g.ndata['nf'] = ...")

            h0 = g.ndata["nf"].to(torch.float32)   # (N, gin_in_dim)
            node_emb = self.gin(g, h0)             # (N, gin_hidden_dim)
            g_emb = self._graph_pool(g, node_emb)  # (B, gin_hidden_dim)

        # --------- Seq encoder ---------
        s_emb = self.lstm(seq, seq_len)        # (B, seq_hidden_dim)

        # --------- Direct concat fusion ---------
        base = torch.cat([g_emb, s_emb], dim=-1)  # (B, D)
        base = self.fuse(base)

        B = base.shape[0]
        T = self.T
        D = base.shape[-1]

        # --------- Build per-task features (复制给每个任务分支) ---------
        # (B, T, D)
        F_task = base.unsqueeze(1).expand(B, T, D).contiguous()

        # --------- Per-task feature extraction (每任务不同提取器) ---------
        F_out = []
        for i in range(T):
            Fi = F_task[:, i, :]                   # (B, D)
            Fi = self.feature_extractors[i](Fi)    # (B, D) 任务i专属
            F_out.append(Fi)
        F_task = torch.stack(F_out, dim=1)         # (B, T, D)

        # --------- Feature sharing across tasks ---------
        F_task = self.feature_sharing(F_task)      # (B, T, D)

        # --------- Per-task ensemble prediction ---------
        values = []
        logits_list = []
        has_logits = False

        for i in range(T):
            Fi = F_task[:, i, :]                   # (B, D)
            # 获取当前任务的 target_bin（如果提供）
            target_bin_i = target_bins[:, i] if target_bins is not None else None
            yi, li = self.ensembles[i](Fi, target_bin=target_bin_i)  # yi: (B, out_dim), li: (B, C) or None
            values.append(yi)
            if li is not None:
                logits_list.append(li)
                has_logits = True

        values = torch.stack(values, dim=1)         # (B, T, out_dim)

        # 如果所有 ensemble 都是 num_classes == 1，则 logits 为 None
        if has_logits and len(logits_list) == T:
            logits = torch.stack(logits_list, dim=1)  # (B, T, C)
        else:
            logits = None

        return values, logits


class TopCircuitSeqModelCfg:
    def __init__(self, num_tasks, gin_in_dim, gin_hidden_dim, gin_layers, 
                 seq_in_dim, seq_hidden_dim, seq_layers, 
                 fe_num_heads, fe_num_layers, fs_num_heads, fs_num_layers, 
                 ens_num_classes, ens_num_layers, ens_hidden_dim, 
                 fs_dropout=0.0, fs_attn_dropout=0.0, ens_dropout=0.0, ens_attn_dropout=0.0,
                 graph_pool="mean"):
        self.num_tasks = num_tasks

        self.gin_in_dim = gin_in_dim
        self.gin_hidden_dim = gin_hidden_dim
        self.gin_layers = gin_layers

        self.seq_in_dim = seq_in_dim
        self.seq_hidden_dim = seq_hidden_dim
        self.seq_layers = seq_layers

        self.task_dim = gin_hidden_dim + seq_hidden_dim

        self.fe_num_heads = fe_num_heads
        self.fe_input_dim = self.task_dim  # 修复: task_dim -> self.task_dim
        self.fe_num_layers = fe_num_layers
        
        self.fs_num_heads = fs_num_heads
        self.fs_num_layers = fs_num_layers
        self.fs_mlp_hidden_dim = self.task_dim  # 修复: task_dim -> self.task_dim

        self.ens_num_classes = ens_num_classes
        self.ens_num_layers = ens_num_layers
        self.ens_hidden_dim = ens_hidden_dim

        self.fs_dropout = fs_dropout
        self.fs_attn_dropout = fs_attn_dropout
        self.ens_dropout = ens_dropout
        self.ens_attn_dropout = ens_attn_dropout
        self.graph_pool = graph_pool


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert DDP-style keys ('module.xxx') to plain keys."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out


def load_pt_checkpoint(
    model: nn.Module,
    pt_path: str,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
    ckpt: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Load weights from a .pt checkpoint into `model`.

    Supported formats:
      1) {"model": state_dict, ...}
      2) {"state_dict": state_dict, ...}
      3) state_dict directly

    If ``ckpt`` is provided (e.g. already loaded via ``torch.load``), it is used
    instead of reading ``pt_path`` again (avoids double disk I/O).
    """
    if ckpt is None:
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Checkpoint not found: {pt_path}")
        ckpt = torch.load(pt_path, map_location=map_location)
    meta: Dict[str, Any] = {}

    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
            meta = {k: v for k, v in ckpt.items() if k != "model"}
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
            meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
        else:
            state_dict = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    state_dict = _strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=strict)
    return meta
        
# -----------------------------
# Quick test (optional)
# -----------------------------
if __name__ == "__main__":
    # dummy test (requires a real DGLGraph with ndata['nf'])
    cfg = TopCircuitSeqModelCfg(
        num_tasks=2,
        gin_in_dim=7,
        gin_hidden_dim=128,
        gin_layers=2,
        seq_in_dim=16,
        seq_hidden_dim=128,
        seq_layers=2,
        fe_num_heads=4,
        fe_num_layers=2,
        fs_num_heads=4,
        fs_num_layers=2,
        ens_num_classes=1,
        ens_num_layers=3,
        ens_hidden_dim=256,
    )
    model = TopCircuitSeqModel(cfg)
    print(model)