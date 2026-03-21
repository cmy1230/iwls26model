"""
LoRA 微调主脚本：加载 checkpoint + 按目标电路过滤数据，微调前后对比指标。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset_loader import CircuitSeqDataset, collate_circuit_seq
from diversity_select import select_diverse_samples
from label_normalizer import LabelNormalizer, compute_metrics_per_circuit
from model import TopCircuitSeqModel, TopCircuitSeqModelCfg, load_pt_checkpoint
from quantile_bins import QuantileBinManager


def _model_cfg_to_dict(cfg: TopCircuitSeqModelCfg) -> Dict[str, Any]:
    return {
        "num_tasks": cfg.num_tasks,
        "gin_in_dim": cfg.gin_in_dim,
        "gin_hidden_dim": cfg.gin_hidden_dim,
        "gin_layers": cfg.gin_layers,
        "seq_in_dim": cfg.seq_in_dim,
        "seq_hidden_dim": cfg.seq_hidden_dim,
        "seq_layers": cfg.seq_layers,
        "fe_num_heads": cfg.fe_num_heads,
        "fe_num_layers": cfg.fe_num_layers,
        "fs_num_heads": cfg.fs_num_heads,
        "fs_num_layers": cfg.fs_num_layers,
        "ens_num_classes": cfg.ens_num_classes,
        "ens_num_layers": cfg.ens_num_layers,
        "ens_hidden_dim": cfg.ens_hidden_dim,
        "fs_dropout": cfg.fs_dropout,
        "fs_attn_dropout": cfg.fs_attn_dropout,
        "ens_dropout": cfg.ens_dropout,
        "ens_attn_dropout": cfg.ens_attn_dropout,
        "graph_pool": cfg.graph_pool,
    }


def _build_run_meta(
    args: argparse.Namespace,
    model_cfg: TopCircuitSeqModelCfg,
    train_indices: Sequence[int],
) -> Dict[str, Any]:
    gcache = (getattr(args, "graph_emb_cache_path", None) or "").strip()
    return {
        "version": 1,
        "circuit_name": args.circuit_name,
        "base_checkpoint": os.path.abspath(args.checkpoint),
        "csv": os.path.abspath(args.csv),
        "circuit_dir": os.path.abspath(args.circuit_dir),
        "seq_dir": os.path.abspath(args.seq_dir),
        "normalizer_path": os.path.abspath(args.normalizer_path) if args.normalizer_path else "",
        "bin_manager_path": os.path.abspath(args.bin_manager_path) if args.bin_manager_path else "",
        "train_n": args.train_n,
        "train_indices": sorted(int(i) for i in train_indices),
        "seed": args.seed,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "eval_full": args.eval_full,
        "graph_emb_cache_path": os.path.abspath(gcache) if gcache else "",
        "no_graph_emb_cache": args.no_graph_emb_cache,
        "gin_hidden_dim": args.gin_hidden_dim,
        "model_cfg": _model_cfg_to_dict(model_cfg),
    }


LABEL_NAMES = ["area", "delay"]


def _extract_labels_from_batch(batch: Dict[str, Any], label_names: List[str], device: str) -> torch.Tensor:
    """从 batch 中提取 label，拼成 (B, T)。与 train.py 逻辑一致。"""
    ys = []
    for lb in label_names:
        if lb not in batch:
            raise KeyError(f"Label '{lb}' not found in batch. Available keys: {list(batch.keys())}")
        y = batch[lb]
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        ys.append(y.to(device).to(torch.float32))
    return torch.stack(ys, dim=1)


def combined_loss(
    values: torch.Tensor,
    logits: Optional[torch.Tensor],
    y_regression: torch.Tensor,
    y_classification: Optional[torch.Tensor],
    alpha: float = 1.0,
    task_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """加权 MSE + alpha * CE；与 train.py 一致。"""
    values = values.squeeze(-1)
    per_sample_mse = (values - y_regression) ** 2
    if task_weights is not None:
        w = task_weights.to(per_sample_mse.device)
        per_sample_mse = per_sample_mse * w.unsqueeze(0)
    mse = per_sample_mse.mean()

    if logits is None or y_classification is None:
        return mse

    T = logits.shape[1]
    ce_per_task = []
    for t in range(T):
        ce_t = nn.CrossEntropyLoss()(
            logits[:, t, :],
            y_classification[:, t].long(),
        )
        ce_per_task.append(ce_t)
    ce_per_task = torch.stack(ce_per_task)
    if task_weights is not None:
        w = task_weights.to(ce_per_task.device)
        ce_loss = (ce_per_task * w).mean()
    else:
        ce_loss = ce_per_task.mean()

    return mse + alpha * ce_loss


def ranking_loss(pred: torch.Tensor, y: torch.Tensor, margin: float = 0.01) -> torch.Tensor:
    """
    Pairwise ranking loss：对 batch 内所有样本对，
    若真值 y_i > y_j，则要求 pred_i > pred_j。
    pred, y: (B, T)，均在同一空间（归一化或原始）。
    """
    diff_y = y.unsqueeze(0) - y.unsqueeze(1)  # (B, B, T)
    diff_p = pred.unsqueeze(0) - pred.unsqueeze(1)  # (B, B, T)
    loss = torch.clamp(margin - torch.sign(diff_y) * diff_p, min=0.0)
    mask = torch.abs(diff_y) > 1e-6  # 只统计真值不同的 pair
    if mask.any():
        return loss[mask].mean()
    return loss.mean()


# -----------------------------
# Block 1: LoRA 模块
# -----------------------------
class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int, lora_alpha: float):
        super().__init__()
        self.original_linear = original_linear
        for p in self.original_linear.parameters():
            p.requires_grad = False

        in_f = original_linear.in_features
        out_f = original_linear.out_features
        # 必须在 original_linear 所在 device/dtype 上创建，否则 model 已 .cuda() 后注入 LoRA 时
        # lora 权重会留在 CPU，forward 会报 cuda/cpu 混用。
        dev = original_linear.weight.device
        dt = original_linear.weight.dtype
        self.lora_A = nn.Linear(in_f, r, bias=False).to(device=dev, dtype=dt)
        self.lora_B = nn.Linear(r, out_f, bias=False).to(device=dev, dtype=dt)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.scaling = lora_alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_linear(x) + self.lora_B(self.lora_A(x)) * self.scaling


def inject_lora(module: nn.Module, r: int, lora_alpha: float) -> None:
    """递归遍历 module：Linear -> LoRALinear；跳过已注入的 LoRALinear 内部。

    注意：不要替换 nn.MultiheadAttention 内部的 Linear。
    PyTorch 的 MultiheadAttention 在 forward 里用 F.linear(x, out_proj.weight, out_proj.bias)
    直接读子模块的 .weight，而 LoRALinear 把权重放在 original_linear 里；若替换 out_proj
    会触发 AttributeError，且即使用 property 转发，也会绕过 LoRALinear.forward，LoRA 不生效。
    """
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        # 保留 MHA 内部结构不变，仅对其外的 Linear 注入 LoRA
        if isinstance(child, nn.MultiheadAttention):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r, lora_alpha))
        else:
            inject_lora(child, r, lora_alpha)


def setup_lora(model: TopCircuitSeqModel, r: int, lora_alpha: float) -> int:
    for p in model.parameters():
        p.requires_grad = False

    inject_lora(model.lstm, r, lora_alpha)  # 适配序列编码器
    inject_lora(model.feature_extractors, r, lora_alpha)
    inject_lora(model.feature_sharing, r, lora_alpha)
    inject_lora(model.ensembles, r, lora_alpha)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[LoRA] trainable parameters: {n_trainable:,}")
    return n_trainable


# -----------------------------
# Block 2: 数据准备
# -----------------------------
def get_circuit_dataset(
    csv_path: str,
    circuit_name: str,
    circuit_dir: str,
    seq_dir: str,
) -> CircuitSeqDataset:
    # 先全表读入并筛行，但不 preload 图，避免为整张 CSV 里所有电路加载 AAG
    ds = CircuitSeqDataset(
        csv_path=csv_path,
        circuit_dir=circuit_dir,
        seq_dir=seq_dir,
        use_header=True,
        check_paths=False,
        preload_graphs=False,
        labels=["area", "delay"],
    )
    col = ds.circuit_col
    mask = ds.df[col].astype(str).str.contains(circuit_name, na=False, regex=False)
    sub = ds.df[mask].reset_index(drop=True)
    if len(sub) == 0:
        raise ValueError(
            f"No rows where '{col}' contains {circuit_name!r}. "
            f"Example values: {ds.df[col].head(5).tolist()}"
        )
    # 仅针对目标电路子集构建 Dataset，graph_pool 只含该电路对应的一个（或少数）AAG
    return CircuitSeqDataset(
        csv_path=csv_path,
        circuit_dir=circuit_dir,
        seq_dir=seq_dir,
        use_header=True,
        check_paths=False,
        preload_graphs=True,
        df=sub,
        labels=["area", "delay"],
    )


def split_train_test(
    circuit_ds: CircuitSeqDataset,
    train_indices: Sequence[int],
) -> Tuple[Subset, Subset]:
    n = len(circuit_ds)
    tr_set: Set[int] = set(int(i) for i in train_indices)
    te_indices = sorted(set(range(n)) - tr_set)
    tr_indices = sorted(tr_set)
    return Subset(circuit_ds, tr_indices), Subset(circuit_ds, te_indices)


def _unwrap_base_dataset(ds: Union[CircuitSeqDataset, Subset]) -> CircuitSeqDataset:
    cur: Any = ds
    while isinstance(cur, Subset):
        cur = cur.dataset
    return cur


@torch.no_grad()
def precompute_gid_g_emb(
    model: TopCircuitSeqModel,
    base_ds: CircuitSeqDataset,
    gids: Sequence[int],
    device: Union[str, torch.device],
) -> Dict[int, torch.Tensor]:
    """对每个 gid 跑一次 GIN+pool，得到 (1, H) 的 CPU tensor，供本脚本内复用。"""
    model.eval()
    out: Dict[int, torch.Tensor] = {}
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    for gid in gids:
        g = base_ds._get_graph(int(gid))
        g = g.to(dev)
        h0 = g.ndata["nf"].to(torch.float32)
        node_emb = model.gin(g, h0)
        ge = model._graph_pool(g, node_emb)
        out[int(gid)] = ge.detach().cpu()
    return out


def batch_g_emb_from_cache(
    batch: Dict[str, Any],
    cache: Dict[int, torch.Tensor],
    device: Union[str, torch.device],
) -> torch.Tensor:
    """按 batch['gid'] 拼出 (B, H)。"""
    gids = batch["gid"]
    if not torch.is_tensor(gids):
        gids = torch.tensor(gids, dtype=torch.long)
    rows: List[torch.Tensor] = []
    for i in range(int(gids.shape[0])):
        gid = int(gids[i].item())
        if gid not in cache:
            raise KeyError(f"gid {gid} missing from g_emb cache (keys={sorted(cache.keys())})")
        rows.append(cache[gid])
    return torch.cat(rows, dim=0).to(device=device, dtype=torch.float32)


def _save_g_emb_cache_file(path: str, cache: Dict[int, torch.Tensor], meta: Dict[str, Any]) -> None:
    torch.save({"meta": meta, "emb": {str(k): v for k, v in cache.items()}}, path)


def _try_load_g_emb_cache_file(
    path: str,
    expect_hidden: int,
    need_gids: Set[int],
) -> Optional[Dict[int, torch.Tensor]]:
    if not path or not os.path.isfile(path):
        return None
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception:
        return None
    emb = obj.get("emb") if isinstance(obj, dict) else None
    if not isinstance(emb, dict):
        return None
    cache = {int(k): v for k, v in emb.items()}
    if not need_gids.issubset(cache.keys()):
        return None
    for gid in need_gids:
        t = cache[gid]
        if t.ndim != 2 or t.shape[-1] != expect_hidden:
            return None
    return cache


# -----------------------------
# Block 3: 训练与评估
# -----------------------------
def train_one_epoch(
    model: TopCircuitSeqModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    normalizer: Optional[LabelNormalizer],
    bin_manager: Optional[QuantileBinManager],
    g_emb_cache: Optional[Dict[int, torch.Tensor]] = None,
) -> float:
    """MSE + 0.5 × CE（有 logits 时）；梯度裁剪 1.0。任务权重与 train.py 一致 (area=1, delay=3)。"""
    model.train()
    total = 0.0
    n_samples = 0
    ce_alpha = 0.5
    task_weights = torch.tensor([1.0, 3.0], dtype=torch.float32)

    for batch in loader:
        bs = batch["seq"].shape[0]
        y = _extract_labels_from_batch(batch, LABEL_NAMES, device)
        y_for_loss = normalizer.normalize(y) if normalizer is not None else y

        if bin_manager is not None and bin_manager.is_fitted:
            y_classification = bin_manager.get_bin_indices(y)
        else:
            y_classification = None

        optimizer.zero_grad(set_to_none=True)
        seq = batch["seq"].to(device)
        seq_len = batch["seq_len"].to(device)
        if g_emb_cache is not None:
            g_emb_b = batch_g_emb_from_cache(batch, g_emb_cache, device)
            out = model(None, seq, seq_len, target_bins=y_classification, g_emb=g_emb_b)
        else:
            g = batch["g"].to(device)
            out = model(g, seq, seq_len, target_bins=y_classification)
        values, logits = out
        loss = combined_loss(
            values,
            logits,
            y_for_loss,
            y_classification,
            alpha=ce_alpha,
            task_weights=task_weights,
        )
        loss = loss + 0.5 * ranking_loss(values.squeeze(-1), y_for_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += float(loss.detach().cpu()) * bs
        n_samples += bs

    return total / max(1, n_samples)


@torch.no_grad()
def forward_collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    normalizer: Optional[LabelNormalizer],
    g_emb_cache: Optional[Dict[int, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回反归一化后的 pred/true 与 gid，供 evaluate 或下游分析复用。"""
    model.eval()
    preds: List[torch.Tensor] = []
    trues: List[torch.Tensor] = []
    gids_list: List[torch.Tensor] = []

    for batch in loader:
        seq = batch["seq"].to(device)
        seq_len = batch["seq_len"].to(device)
        y = _extract_labels_from_batch(batch, LABEL_NAMES, device)
        if g_emb_cache is not None:
            g_emb_b = batch_g_emb_from_cache(batch, g_emb_cache, device)
            out = model(None, seq, seq_len, target_bins=None, g_emb=g_emb_b)
        else:
            g = batch["g"].to(device)
            out = model(g, seq, seq_len, target_bins=None)
        p = out[0] if isinstance(out, (tuple, list)) else out
        if p.ndim == 3 and p.shape[-1] == 1:
            p = p.squeeze(-1)
        if normalizer is not None:
            p = normalizer.denormalize(p)
        preds.append(p.detach().cpu())
        trues.append(y.detach().cpu())
        gids_list.append(batch["gid"].detach().cpu())

    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)
    gids = torch.cat(gids_list, dim=0)
    return pred, true, gids


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    normalizer: Optional[LabelNormalizer],
    g_emb_cache: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[str, Any]:
    pred, true, gids = forward_collect_predictions(
        model, loader, device, normalizer, g_emb_cache=g_emb_cache
    )
    return compute_metrics_per_circuit(pred, true, gids, LABEL_NAMES)


# -----------------------------
# utils
# -----------------------------
def _build_model_cfg(args: argparse.Namespace, num_tasks: int) -> TopCircuitSeqModelCfg:
    return TopCircuitSeqModelCfg(
        num_tasks=num_tasks,
        gin_in_dim=args.gin_in_dim,
        gin_hidden_dim=args.gin_hidden_dim,
        gin_layers=args.gin_layers,
        seq_in_dim=args.seq_in_dim,
        seq_hidden_dim=args.seq_hidden_dim,
        seq_layers=args.seq_layers,
        fe_num_heads=args.fe_num_heads,
        fe_num_layers=args.fe_num_layers,
        fs_num_heads=args.fs_num_heads,
        fs_num_layers=args.fs_num_layers,
        ens_num_classes=args.ens_num_classes,
        ens_num_layers=args.ens_num_layers,
        ens_hidden_dim=args.ens_hidden_dim,
        fs_dropout=args.fs_dropout,
        fs_attn_dropout=args.fs_attn_dropout,
        ens_dropout=args.ens_dropout,
        ens_attn_dropout=args.ens_attn_dropout,
        graph_pool=args.graph_pool,
    )


def _metrics_to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _metrics_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_metrics_to_jsonable(x) for x in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, float):
        return obj
    return obj


def _collect_train_labels(train_ds: Subset, label_names: List[str]) -> np.ndarray:
    rows = []
    base = train_ds.dataset
    for i in train_ds.indices:
        row = base.df.iloc[i]
        rows.append([float(row[base.label2col[lb]]) for lb in label_names])
    return np.asarray(rows, dtype=np.float64)


# -----------------------------
# Block 4: main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA finetune on a single circuit (subset)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python finetune_lora.py \\
    --checkpoint ckpt/model.pt \\
    --csv data/train.csv \\
    --circuit_name my_design \\
    --circuit_dir ./graphs \\
    --seq_dir ./seqs \\
    --train_n 32 \\
    --normalizer_path ckpt/label_norm.json \\
    --bin_manager_path ckpt/quantile_bins.json

说明:
  --circuit_name 在 CSV 的电路名列中做子串匹配, 筛出该电路所有行;
  --train_n 为多样性采样得到的训练条数, 其余行作为 test;
  模型结构参数需与预训练 checkpoint 一致 (脚本会从 ckpt 读取 num_bins 等)。
  默认对每个 gid 预计算 GIN+pool 图向量并复用（大幅减轻 CPU 上重复大图前向）；
  可用 --graph_emb_cache_path 保存/加载 .pt，或 --no_graph_emb_cache 关闭。
  加 --eval_full 可在该电路全部 CSV 行上额外算 pooled R² / MAPE / Pearson（含曾用于训练的行）。
""",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--circuit_name", type=str, required=True)
    parser.add_argument("--circuit_dir", type=str, required=True)
    parser.add_argument("--seq_dir", type=str, required=True)
    parser.add_argument("--normalizer_path", type=str, default="")
    parser.add_argument("--bin_manager_path", type=str, default="",
                        help="可选；若不提供则在 train 子集上 fit QuantileBinManager")
    parser.add_argument("--train_n", type=int, required=True, help="多样性选中的训练样本数")
    parser.add_argument("--seed", type=int, default=42, help="多样性选样本的随机种子")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./lora_finetune_out")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--no_cudnn",
        action="store_true",
        help="关闭 cuDNN 卷积（用原生 CUDA 实现，略慢），可规避 "
        "'Unable to find a valid cuDNN algorithm to run convolution'",
    )

    # 模型结构（需与 checkpoint 一致）
    parser.add_argument("--gin_in_dim", type=int, default=8)
    parser.add_argument("--gin_hidden_dim", type=int, default=128)
    parser.add_argument("--gin_layers", type=int, default=2)
    parser.add_argument("--seq_in_dim", type=int, default=31)
    parser.add_argument("--seq_hidden_dim", type=int, default=128)
    parser.add_argument("--seq_layers", type=int, default=2)
    parser.add_argument("--fe_num_heads", type=int, default=4)
    parser.add_argument("--fe_num_layers", type=int, default=2)
    parser.add_argument("--fs_num_heads", type=int, default=4)
    parser.add_argument("--fs_num_layers", type=int, default=2)
    parser.add_argument("--ens_num_classes", type=int, default=8)
    parser.add_argument("--ens_num_layers", type=int, default=3)
    parser.add_argument("--ens_hidden_dim", type=int, default=256)
    parser.add_argument("--fs_dropout", type=float, default=0.0)
    parser.add_argument("--fs_attn_dropout", type=float, default=0.0)
    parser.add_argument("--ens_dropout", type=float, default=0.0)
    parser.add_argument("--ens_attn_dropout", type=float, default=0.0)
    parser.add_argument("--graph_pool", type=str, default="mean")
    parser.add_argument(
        "--no_graph_emb_cache",
        action="store_true",
        help="禁用按 gid 预计算并缓存 GIN+pool 图向量（每步重跑整张图，较慢）",
    )
    parser.add_argument(
        "--graph_emb_cache_path",
        type=str,
        default="",
        help="可选：.pt 路径。若存在且含当前电路全部 gid 则直接加载；否则预计算后写入。",
    )
    parser.add_argument(
        "--eval_full",
        action="store_true",
        help="在该电路 CSV 全部样本上额外评估（含训练行；R²/MAPE/Pearson 为全样本 pooled，见 metrics_*_full）",
    )

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    num_tasks = len(LABEL_NAMES)

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        if args.no_cudnn:
            torch.backends.cudnn.enabled = False
            print("[cuda] cudnn.enabled=False (--no_cudnn)，避免部分环境下 cuDNN 选算法失败")
        else:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)

    # 数据集
    circuit_ds = get_circuit_dataset(
        csv_path=args.csv,
        circuit_name=args.circuit_name,
        circuit_dir=args.circuit_dir,
        seq_dir=args.seq_dir,
    )
    seq_paths = [str(circuit_ds.df.iloc[i][circuit_ds.seq_col]) for i in range(len(circuit_ds))]
    train_idx = select_diverse_samples(seq_paths, args.train_n, seed=args.seed)
    train_ds, test_ds = split_train_test(circuit_ds, train_idx)

    print(
        f"[data] circuit={args.circuit_name} total={len(circuit_ds)} "
        f"train={len(train_ds)} test={len(test_ds)}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_circuit_seq,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_circuit_seq,
    )
    full_loader: Optional[DataLoader] = None
    if args.eval_full:
        full_loader = DataLoader(
            circuit_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_circuit_seq,
        )

    # Normalizer
    normalizer: Optional[LabelNormalizer] = None
    if args.normalizer_path:
        normalizer = LabelNormalizer.load(args.normalizer_path)

    # 先从 checkpoint 读 meta，便于覆盖 num_bins 等
    ckpt_raw = torch.load(args.checkpoint, map_location="cpu")
    meta: Dict[str, Any] = ckpt_raw if isinstance(ckpt_raw, dict) else {}
    if isinstance(ckpt_raw, dict) and "num_bins" in ckpt_raw:
        args.ens_num_classes = max(1, int(ckpt_raw["num_bins"]))
    if isinstance(ckpt_raw, dict) and ckpt_raw.get("use_quantile_bins") is False:
        args.ens_num_classes = 1

    model_cfg = _build_model_cfg(args, num_tasks=num_tasks)
    model = TopCircuitSeqModel(model_cfg).to(device)
    try:
        load_pt_checkpoint(model, args.checkpoint, map_location=device, ckpt=ckpt_raw)
    except TypeError:
        load_pt_checkpoint(model, args.checkpoint, map_location=device)
    print(f"[model] loaded on {device!r}", flush=True)

    g_emb_cache: Optional[Dict[int, torch.Tensor]] = None
    if not args.no_graph_emb_cache:
        base_ds = _unwrap_base_dataset(circuit_ds)
        need_gids = {int(x) for x in circuit_ds.df["gid"].tolist()}
        gids_sorted = sorted(need_gids)
        cache_path = (args.graph_emb_cache_path or "").strip()
        loaded: Optional[Dict[int, torch.Tensor]] = None
        if cache_path:
            loaded = _try_load_g_emb_cache_file(cache_path, args.gin_hidden_dim, need_gids)
        if loaded is not None:
            g_emb_cache = loaded
            print(f"[g_emb] loaded from {cache_path} ({len(g_emb_cache)} gids)", flush=True)
        else:
            print(f"[g_emb] precomputing GIN+pool for gids={gids_sorted} ...", flush=True)
            t0 = time.perf_counter()
            g_emb_cache = precompute_gid_g_emb(model, base_ds, gids_sorted, device)
            print(
                f"[g_emb] precompute done in {time.perf_counter() - t0:.2f}s (CPU tensors)",
                flush=True,
            )
            if cache_path:
                _save_g_emb_cache_file(
                    cache_path,
                    g_emb_cache,
                    {
                        "checkpoint": os.path.abspath(args.checkpoint),
                        "gin_hidden_dim": args.gin_hidden_dim,
                        "gids": gids_sorted,
                    },
                )
                print(f"[g_emb] saved to {cache_path}", flush=True)
    else:
        print("[g_emb] cache disabled (--no_graph_emb_cache)", flush=True)

    # -------- 微调前（基座 checkpoint）--------
    n_te = len(test_ds)
    n_bt = (n_te + args.batch_size - 1) // max(1, args.batch_size)
    print(
        f"[eval] baseline on test: {n_te} samples, ~{n_bt} batches (CPU 上可能很慢，请耐心等待)",
        flush=True,
    )
    metrics_before = evaluate(model, test_loader, device, normalizer, g_emb_cache=g_emb_cache)
    print("[eval] baseline done", flush=True)
    metrics_before_full: Optional[Dict[str, Any]] = None
    if full_loader is not None:
        n_f = len(circuit_ds)
        print(f"[eval] baseline on FULL ({n_f} samples, 含训练行)...", flush=True)
        metrics_before_full = evaluate(model, full_loader, device, normalizer, g_emb_cache=g_emb_cache)
        print("[eval] baseline full done", flush=True)

    # LoRA（注入后再次 .to(device)，与 LoRALinear 内 device 对齐形成双保险）
    setup_lora(model, args.lora_r, args.lora_alpha)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    # Bin manager：teacher forcing
    bin_manager: Optional[QuantileBinManager] = None
    if args.bin_manager_path:
        bin_manager = QuantileBinManager.load(args.bin_manager_path)
    else:
        if model_cfg.ens_num_classes > 1:
            lab = _collect_train_labels(train_ds, LABEL_NAMES)
            bin_manager = QuantileBinManager(num_tasks=num_tasks, num_bins=model_cfg.ens_num_classes)
            bin_manager.fit(lab)
            p = os.path.join(args.output_dir, "quantile_bins_finetune_fit.json")
            bin_manager.save(p)
            print(f"[bin] fitted on train subset, saved {p}")

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model, train_loader, opt, device, normalizer, bin_manager, g_emb_cache=g_emb_cache
        )
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            print(f"[epoch {epoch}/{args.epochs}] train_loss={loss:.6f}")

    print("[eval] after finetune on test...", flush=True)
    metrics_after = evaluate(model, test_loader, device, normalizer, g_emb_cache=g_emb_cache)
    print("[eval] after finetune done", flush=True)
    metrics_after_full: Optional[Dict[str, Any]] = None
    if full_loader is not None:
        print(f"[eval] after finetune on FULL ({len(circuit_ds)} samples)...", flush=True)
        metrics_after_full = evaluate(model, full_loader, device, normalizer, g_emb_cache=g_emb_cache)
        print("[eval] after finetune full done", flush=True)

    run_meta = _build_run_meta(args, model_cfg, train_idx)
    lora_pt = os.path.join(args.output_dir, "lora_finetuned_model.pt")
    torch.save(
        {"state_dict": model.state_dict(), "meta": run_meta},
        lora_pt,
    )
    meta_json_path = os.path.join(args.output_dir, "lora_run_meta.json")
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(_metrics_to_jsonable(run_meta), f, indent=2, ensure_ascii=False)
    print(f"[done] saved LoRA checkpoint: {lora_pt}", flush=True)
    print(f"[done] saved run meta: {meta_json_path}", flush=True)

    out = {
        "circuit_name": args.circuit_name,
        "checkpoint": args.checkpoint,
        "train_n": args.train_n,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "graph_emb_cache": g_emb_cache is not None,
        "lora_finetuned_model_path": os.path.abspath(lora_pt),
        "lora_run_meta_path": os.path.abspath(meta_json_path),
        "metrics_before": _metrics_to_jsonable(metrics_before),
        "metrics_after": _metrics_to_jsonable(metrics_after),
    }
    if metrics_before_full is not None:
        out["metrics_before_full"] = _metrics_to_jsonable(metrics_before_full)
    if metrics_after_full is not None:
        out["metrics_after_full"] = _metrics_to_jsonable(metrics_after_full)
    out_path = os.path.join(args.output_dir, "lora_finetune_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[done] metrics saved to {out_path}")

    def _summarize(tag: str, m: Dict[str, Any]) -> None:
        r2 = m.get("r2_avg_over_circuits", m.get("r2_mean", 0.0))
        mape = m.get("mape_avg_over_circuits", m.get("mape_mean", 0.0))
        print(f"  [{tag}] R2_avg_over_circuits={r2:.4f} MAPE_avg_over_circuits={mape*100:.2f}% mse={m.get('mse', 0):.6f}")
        r2_pt = m.get("r2_per_task") or []
        mape_pt = m.get("mape_per_task") or []
        pr_pt = m.get("pearson_per_task") or []
        names = m.get("label_names") or LABEL_NAMES
        for i, name in enumerate(names):
            if i < len(r2_pt) and i < len(mape_pt):
                pr_s = f" Pearson={pr_pt[i]:.4f}" if i < len(pr_pt) else ""
                print(
                    f"       {name}: R2={r2_pt[i]:.4f} MAPE={mape_pt[i]*100:.2f}%{pr_s}"
                )

    print("\n=== 微调前后对比（test 子集）===")
    _summarize("before", metrics_before)
    _summarize("after ", metrics_after)
    if metrics_before_full is not None and metrics_after_full is not None:
        print("\n=== 全量样本（含训练行）pooled 指标 ===")
        _summarize("before_full", metrics_before_full)
        _summarize("after_full ", metrics_after_full)


if __name__ == "__main__":
    main()
