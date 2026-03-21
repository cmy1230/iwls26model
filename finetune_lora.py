"""
LoRA 微调主脚本：加载 checkpoint + 按目标电路过滤数据，微调前后对比指标。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset_loader import CircuitSeqDataset, collate_circuit_seq
from diversity_select import select_diverse_samples
from label_normalizer import LabelNormalizer, compute_metrics_per_circuit
from model import TopCircuitSeqModel, TopCircuitSeqModelCfg, load_pt_checkpoint
from quantile_bins import QuantileBinManager

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
        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.scaling = lora_alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_linear(x) + self.lora_B(self.lora_A(x)) * self.scaling


def inject_lora(module: nn.Module, r: int, lora_alpha: float) -> None:
    """递归遍历 module：Linear -> LoRALinear；跳过已注入的 LoRALinear 内部。"""
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r, lora_alpha))
        else:
            inject_lora(child, r, lora_alpha)


def setup_lora(model: TopCircuitSeqModel, r: int, lora_alpha: float) -> int:
    for p in model.parameters():
        p.requires_grad = False

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
    ds = CircuitSeqDataset(
        csv_path=csv_path,
        circuit_dir=circuit_dir,
        seq_dir=seq_dir,
        use_header=True,
        check_paths=False,
        preload_graphs=True,
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
    return ds.make_subset(sub)


def split_train_test(
    circuit_ds: CircuitSeqDataset,
    train_indices: Sequence[int],
) -> Tuple[Subset, Subset]:
    n = len(circuit_ds)
    tr_set: Set[int] = set(int(i) for i in train_indices)
    te_indices = sorted(set(range(n)) - tr_set)
    tr_indices = sorted(tr_set)
    return Subset(circuit_ds, tr_indices), Subset(circuit_ds, te_indices)


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
        g = batch["g"].to(device)
        seq = batch["seq"].to(device)
        seq_len = batch["seq_len"].to(device)
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
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += float(loss.detach().cpu()) * bs
        n_samples += bs

    return total / max(1, n_samples)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    normalizer: Optional[LabelNormalizer],
) -> Dict[str, Any]:
    model.eval()
    preds: List[torch.Tensor] = []
    trues: List[torch.Tensor] = []
    gids_list: List[torch.Tensor] = []

    for batch in loader:
        g = batch["g"].to(device)
        seq = batch["seq"].to(device)
        seq_len = batch["seq_len"].to(device)
        y = _extract_labels_from_batch(batch, LABEL_NAMES, device)
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
    parser = argparse.ArgumentParser(description="LoRA finetune on a single circuit (subset)")
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

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    num_tasks = len(LABEL_NAMES)

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
    load_pt_checkpoint(model, args.checkpoint, map_location=device)

    # -------- 微调前（基座 checkpoint）--------
    metrics_before = evaluate(model, test_loader, device, normalizer)

    # LoRA
    setup_lora(model, args.lora_r, args.lora_alpha)
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
        loss = train_one_epoch(model, train_loader, opt, device, normalizer, bin_manager)
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            print(f"[epoch {epoch}/{args.epochs}] train_loss={loss:.6f}")

    metrics_after = evaluate(model, test_loader, device, normalizer)

    out = {
        "circuit_name": args.circuit_name,
        "checkpoint": args.checkpoint,
        "train_n": args.train_n,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "metrics_before": _metrics_to_jsonable(metrics_before),
        "metrics_after": _metrics_to_jsonable(metrics_after),
    }
    out_path = os.path.join(args.output_dir, "lora_finetune_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[done] metrics saved to {out_path}")

    def _summarize(tag: str, m: Dict[str, Any]) -> None:
        r2 = m.get("r2_avg_over_circuits", m.get("r2_mean", 0.0))
        mape = m.get("mape_avg_over_circuits", m.get("mape_mean", 0.0))
        print(f"  [{tag}] R2_avg_over_circuits={r2:.4f} MAPE_avg_over_circuits={mape*100:.2f}% mse={m.get('mse', 0):.6f}")

    print("\n=== 微调前后对比（test 子集）===")
    _summarize("before", metrics_before)
    _summarize("after ", metrics_after)


if __name__ == "__main__":
    main()
