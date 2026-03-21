# label_normalizer.py
"""
Label Normalizer (log1p + z-score) for multi-task regression.

What this file provides:
  - LabelNormalizer: fit stats from TRAIN set, then (log1p -> zscore) normalize / denormalize.
  - NormalizedMSELoss: plain MSE between normalized pred/target.
  - compute_metrics_original_space: MSE/R2/MAPE/MAE in ORIGINAL space.

Default transform:
  y_log = log1p(y)            # (optional per-label, recommended for long-tail positive labels)
  y_norm = (y_log - mean)/std # mean/std computed on y_log from train set
  denorm: y = expm1(y_norm*std + mean)

Notes:
  - log1p requires y >= 0. If a label may be negative, disable log for that label.
  - mean/std are computed on TRAIN only.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn as nn

from dataset_loader import CircuitSeqDataset


@dataclass
class LabelStats:
    """Statistics for a single label (computed in transformed space)."""
    name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    count: int
    use_log1p: bool = True  # whether this label uses log1p/expm1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mean": self.mean,
            "std": self.std,
            "min": self.min_val,
            "max": self.max_val,
            "count": self.count,
            "use_log1p": self.use_log1p,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LabelStats":
        return cls(
            name=d["name"],
            mean=float(d["mean"]),
            std=float(d["std"]),
            min_val=float(d["min"]),
            max_val=float(d["max"]),
            count=int(d["count"]),
            use_log1p=bool(d.get("use_log1p", True)),
        )


class LabelNormalizer:
    """
    Per-label (log1p -> zscore) normalizer.

    transform:
      if use_log1p[label]:
          t = log1p(y)
      else:
          t = y
      t_norm = (t - mean) / std

    inverse:
      t = t_norm * std + mean
      if use_log1p[label]:
          y = expm1(t)
      else:
          y = t
    """

    def __init__(
        self,
        labels: List[str],
        eps: float = 1e-8,
        use_log1p: Union[bool, Sequence[str], Set[str], Dict[str, bool]] = True,
        clamp_min_before_log: float = 0.0,
    ):
        """
        Args:
            labels: label names in order.
            eps: avoid division by zero.
            use_log1p:
                - True: apply log1p to ALL labels
                - False: apply log1p to NONE
                - list/set of label names: apply log1p only to these labels
                - dict: per-label boolean
            clamp_min_before_log: before log1p, do y = clamp(y, min=clamp_min_before_log)
                                 (helps if tiny negative noise exists, but ideally labels are >=0)
        """
        self.labels = list(labels)
        self.eps = float(eps)
        self.clamp_min_before_log = float(clamp_min_before_log)

        self._use_log_map: Dict[str, bool] = self._parse_use_log1p(use_log1p)

        self.stats: Dict[str, LabelStats] = {}
        self._mean: Optional[torch.Tensor] = None  # (T,)
        self._std: Optional[torch.Tensor] = None   # (T,)

    def _parse_use_log1p(self, use_log1p) -> Dict[str, bool]:
        if isinstance(use_log1p, bool):
            return {lb: use_log1p for lb in self.labels}
        if isinstance(use_log1p, dict):
            return {lb: bool(use_log1p.get(lb, False)) for lb in self.labels}
        # sequence or set: listed labels use log
        s = set(use_log1p)
        return {lb: (lb in s) for lb in self.labels}

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    @property
    def is_fitted(self) -> bool:
        return len(self.stats) == len(self.labels) and self._mean is not None and self._std is not None

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("LabelNormalizer is not fitted. Call compute_stats() first.")

    def _apply_transform_1d(self, x: torch.Tensor, lb: str) -> torch.Tensor:
        """
        Apply optional log1p transform on a 1D tensor.
        Stats are computed in this transformed space.
        """
        if self._use_log_map.get(lb, False):
            # ensure non-negative before log1p
            if self.clamp_min_before_log is not None:
                x = torch.clamp(x, min=self.clamp_min_before_log)
            # log1p
            x = torch.log1p(x)
        return x

    def _apply_inverse_1d(self, t: torch.Tensor, lb: str) -> torch.Tensor:
        """
        Inverse of transform in transformed space.
        """
        if self._use_log_map.get(lb, False):
            t = torch.expm1(t)
        return t

    @torch.no_grad()
    def compute_stats(self, ds: CircuitSeqDataset) -> "LabelNormalizer":
        """
        Compute stats from ds.df columns for each label.
        IMPORTANT: stats are computed on TRANSFORMED values (log1p if enabled).

        Args:
            ds: typically train_ds

        Returns:
            self
        """
        if len(ds) == 0:
            raise ValueError("empty dataset")

        for lb in self.labels:
            if lb not in ds.df.columns:
                raise KeyError(f"'{lb}' not in ds.df columns: {list(ds.df.columns)}")

            raw = torch.tensor(ds.df[lb].astype(float).values, dtype=torch.float32)

            # transform for stats
            t = self._apply_transform_1d(raw, lb)

            std = float(t.std(unbiased=False).item())
            self.stats[lb] = LabelStats(
                name=lb,
                mean=float(t.mean().item()),
                std=std,
                min_val=float(t.min().item()),
                max_val=float(t.max().item()),
                count=int(t.numel()),
                use_log1p=bool(self._use_log_map.get(lb, False)),
            )

        self._build_tensors()
        return self

    def _build_tensors(self):
        means = []
        stds = []
        for lb in self.labels:
            s = self.stats[lb]
            means.append(float(s.mean))
            stds.append(max(float(s.std), self.eps))
        self._mean = torch.tensor(means, dtype=torch.float32)
        self._std = torch.tensor(stds, dtype=torch.float32)

    def _reshape_for_broadcast(self, v: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        v: (T,) -> broadcast to y's shape.
        y can be (B,T) or (B,T,1).
        """
        if y.ndim == 2:
            return v.view(1, -1)
        if y.ndim == 3 and y.shape[-1] == 1:
            return v.view(1, -1, 1)
        raise ValueError(f"Unsupported y shape for broadcast: {tuple(y.shape)}")

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        """
        Normalize ORIGINAL-space labels to normalized space.

        Args:
            y: (B,T) or (B,T,1) in ORIGINAL space

        Returns:
            y_norm: same shape, in normalized space
        """
        self._check_fitted()

        if y.ndim not in (2, 3):
            raise ValueError(f"y must be (B,T) or (B,T,1); got {tuple(y.shape)}")

        # apply per-label transform
        # easier: operate in (B,T) then restore
        squeeze_last = (y.ndim == 3 and y.shape[-1] == 1)
        y2 = y.squeeze(-1) if squeeze_last else y  # (B,T)

        # transform each column
        cols = []
        for i, lb in enumerate(self.labels):
            xi = y2[:, i]
            xi = self._apply_transform_1d(xi, lb)
            cols.append(xi)
        t = torch.stack(cols, dim=1)  # (B,T)

        mean = self._mean.to(t.device)
        std = self._std.to(t.device)
        t_norm = (t - mean) / std

        if squeeze_last:
            t_norm = t_norm.unsqueeze(-1)
        return t_norm

    def denormalize(self, y_norm: torch.Tensor) -> torch.Tensor:
        """
        Denormalize normalized predictions back to ORIGINAL space.

        Args:
            y_norm: (B,T) or (B,T,1) in normalized space

        Returns:
            y: same shape, in ORIGINAL space
        """
        self._check_fitted()

        if y_norm.ndim not in (2, 3):
            raise ValueError(f"y_norm must be (B,T) or (B,T,1); got {tuple(y_norm.shape)}")

        squeeze_last = (y_norm.ndim == 3 and y_norm.shape[-1] == 1)
        yn = y_norm.squeeze(-1) if squeeze_last else y_norm  # (B,T)

        mean = self._mean.to(yn.device)
        std = self._std.to(yn.device)
        t = yn * std + mean  # back to transformed space

        # inverse per label
        cols = []
        for i, lb in enumerate(self.labels):
            ti = t[:, i]
            yi = self._apply_inverse_1d(ti, lb)
            cols.append(yi)
        y = torch.stack(cols, dim=1)  # (B,T)

        if squeeze_last:
            y = y.unsqueeze(-1)
        return y

    def get_mean_std_tensors(self, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mean/std in transformed space (after log1p if enabled).
        Useful if you want to register buffers elsewhere.
        """
        self._check_fitted()
        return self._mean.to(device), self._std.to(device)

    def print_stats(self):
        self._check_fitted()
        print(f"LabelNormalizer stats ({len(self.labels)} labels, {self.stats[self.labels[0]].count} samples):")
        print(f"  transform: log1p enabled labels = {[lb for lb in self.labels if self._use_log_map.get(lb, False)]}")
        for lb in self.labels:
            s = self.stats[lb]
            print(
                f"  {lb:>8s}: "
                f"use_log1p={s.use_log1p} "
                f"mean={s.mean:12.6f}  std={s.std:12.6f}  "
                f"min={s.min_val:12.6f}  max={s.max_val:12.6f}"
            )

    def save(self, path: str):
        self._check_fitted()
        data = {
            "labels": self.labels,
            "eps": self.eps,
            "clamp_min_before_log": self.clamp_min_before_log,
            "use_log1p": {lb: bool(self._use_log_map.get(lb, False)) for lb in self.labels},
            "stats": {lb: self.stats[lb].to_dict() for lb in self.labels},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[LabelNormalizer] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LabelNormalizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        labels = list(data["labels"])
        normalizer = cls(
            labels=labels,
            eps=float(data.get("eps", 1e-8)),
            use_log1p=data.get("use_log1p", True),
            clamp_min_before_log=float(data.get("clamp_min_before_log", 0.0)),
        )
        for lb in normalizer.labels:
            normalizer.stats[lb] = LabelStats.from_dict(data["stats"][lb])
        normalizer._build_tensors()
        print(f"[LabelNormalizer] Loaded from {path}")
        return normalizer


# -----------------------------
# Normalized Loss Wrapper
# -----------------------------
class NormalizedMSELoss(nn.Module):
    """
    Plain MSE in normalized space.
    You should pass pred_norm and target_norm (both already normalized).
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_norm: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
        if pred_norm.ndim == 3 and pred_norm.shape[-1] == 1:
            pred_norm = pred_norm.squeeze(-1)
        if target_norm.ndim == 3 and target_norm.shape[-1] == 1:
            target_norm = target_norm.squeeze(-1)

        mse = (pred_norm - target_norm) ** 2
        if self.reduction == "mean":
            return mse.mean()
        if self.reduction == "sum":
            return mse.sum()
        return mse


# -----------------------------
# Metrics in Original Space
# -----------------------------
@torch.no_grad()
def compute_metrics_original_space(
    pred: torch.Tensor,
    target: torch.Tensor,
    label_names: List[str],
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """
    Compute metrics (MSE, R², MAPE, MAE) in ORIGINAL space.
    pred/target: (N,T) or (N,T,1)
    """
    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    if target.ndim == 3 and target.shape[-1] == 1:
        target = target.squeeze(-1)

    mse = ((pred - target) ** 2).mean().item()

    r2_list, mape_list, mae_list = [], [], []
    num_tasks = target.shape[1]

    for t in range(num_tasks):
        yt = target[:, t]
        pt = pred[:, t]

        # R²
        sse = ((yt - pt) ** 2).sum()
        ymean = yt.mean()
        sst = ((yt - ymean) ** 2).sum()
        if sst.item() < eps:
            r2 = 0.0
        else:
            r2 = float((1.0 - sse / (sst + eps)).item())
        r2_list.append(r2)

        # MAPE
        denom = torch.clamp(torch.abs(yt), min=eps)
        mape = (torch.abs(yt - pt) / denom).mean().item()
        mape_list.append(mape)

        # MAE
        mae = torch.abs(yt - pt).mean().item()
        mae_list.append(mae)

    r2_mean = sum(r2_list) / len(r2_list)
    mape_mean = sum(mape_list) / len(mape_list)
    mae_mean = sum(mae_list) / len(mae_list)

    return {
        "mse": mse,
        "mae_mean": mae_mean,
        "mae_per_task": mae_list,
        "r2_mean": r2_mean,
        "r2_per_task": r2_list,
        "mape_mean": mape_mean,
        "mape_per_task": mape_list,
        "n_samples": int(target.shape[0]),
        "label_names": label_names,
    }


@torch.no_grad()
def compute_metrics_per_circuit(
    pred: torch.Tensor,
    target: torch.Tensor,
    gids: torch.Tensor,
    label_names: List[str],
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """
    按电路分组计算 R² 和 MAPE（电路内部），返回每个电路的指标及电路间平均值。

    pred/target: (N, T)
    gids: (N,) 每个样本所属的电路 id

    Returns:
        - r2_per_circuit: Dict[gid, List[float]] 每个电路每个任务的 R²（电路内）
        - r2_mean_per_circuit: Dict[gid, float] 每个电路跨任务的平均 R²
        - r2_avg_over_circuits: float 所有电路 r2_mean 的平均，作为训练目标
        - mape_per_circuit: Dict[gid, List[float]] 每个电路每个任务的 MAPE（电路内）
        - mape_mean_per_circuit: Dict[gid, float] 每个电路跨任务的平均 MAPE
        - mape_avg_over_circuits: float 所有电路 mape_mean 的平均
        - 以及 compute_metrics_original_space 的常规指标
    """
    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    if target.ndim == 3 and target.shape[-1] == 1:
        target = target.squeeze(-1)

    gids_t = gids if torch.is_tensor(gids) else torch.tensor(gids, dtype=torch.long)
    if gids_t.device != pred.device:
        gids_t = gids_t.to(pred.device)
    unique_gids = sorted(set(int(g) for g in gids_t.cpu().tolist()))

    r2_per_circuit: Dict[int, List[float]] = {}
    r2_mean_per_circuit: Dict[int, float] = {}
    mape_per_circuit: Dict[int, List[float]] = {}
    mape_mean_per_circuit: Dict[int, float] = {}
    num_tasks = target.shape[1]

    for gid in unique_gids:
        mask = (gids_t == gid)
        pred_g = pred[mask]  # (n_g, T)
        true_g = target[mask]  # (n_g, T)
        n_g = pred_g.shape[0]

        r2_list = []
        mape_list = []
        for t in range(num_tasks):
            yt = true_g[:, t]
            pt = pred_g[:, t]
            # R² (电路内)
            sse = ((yt - pt) ** 2).sum()
            ymean = yt.mean()
            sst = ((yt - ymean) ** 2).sum()
            if sst.item() < eps or n_g < 2:
                r2 = 0.0
            else:
                r2 = float((1.0 - sse / (sst + eps)).item())
            r2_list.append(r2)
            # MAPE (电路内)
            denom = torch.clamp(torch.abs(yt), min=eps)
            mape = float((torch.abs(yt - pt) / denom).mean().item())
            mape_list.append(mape)

        r2_per_circuit[gid] = r2_list
        r2_mean_per_circuit[gid] = sum(r2_list) / len(r2_list)
        mape_per_circuit[gid] = mape_list
        mape_mean_per_circuit[gid] = sum(mape_list) / len(mape_list)

    # 电路间 R² 平均值（训练目标）
    if len(r2_mean_per_circuit) > 0:
        r2_avg_over_circuits = sum(r2_mean_per_circuit.values()) / len(r2_mean_per_circuit)
    else:
        r2_avg_over_circuits = 0.0

    # 电路间 MAPE 平均值
    if len(mape_mean_per_circuit) > 0:
        mape_avg_over_circuits = sum(mape_mean_per_circuit.values()) / len(mape_mean_per_circuit)
    else:
        mape_avg_over_circuits = 0.0

    # 常规整体指标（保持兼容）
    base = compute_metrics_original_space(pred, target, label_names, eps)

    base["r2_per_circuit"] = r2_per_circuit
    base["r2_mean_per_circuit"] = r2_mean_per_circuit
    base["r2_avg_over_circuits"] = r2_avg_over_circuits
    base["mape_per_circuit"] = mape_per_circuit
    base["mape_mean_per_circuit"] = mape_mean_per_circuit
    base["mape_avg_over_circuits"] = mape_avg_over_circuits
    return base


# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    from split_dataset import split_dataset

    # Build dataset
    ds = CircuitSeqDataset(
        csv_path="/home/yfdai/asap/data/data.csv",
        circuit_dir="/home/yfdai/asap/data/aig",
        seq_dir="/home/yfdai/asap/data/seq/",
        use_header=True,
        preload_graphs=True,
        labels=["nd", "area", "delay", "lev", "runtime"],
    )

    train_ds, val_ds, test_ds = split_dataset(
        ds, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
        mode="within_circuit", seed=0,
    )

    # Use log1p for long-tail positive labels; you can customize here:
    # - True: all labels use log1p
    # - ["nd","area","delay"]: only these use log1p
    normalizer = LabelNormalizer(labels=ds.labels, use_log1p=["nd", "area", "delay"], clamp_min_before_log=0.0)
    normalizer.compute_stats(train_ds)
    normalizer.print_stats()

    # Save & reload
    normalizer.save("./label_normalizer_stats.json")
    normalizer2 = LabelNormalizer.load("./label_normalizer_stats.json")
    normalizer2.print_stats()

    # Test normalize/denormalize on one sample
    sample = train_ds[0]
    y = torch.stack([sample.y[lb] for lb in ds.labels], dim=0).unsqueeze(0)  # (1, T)
    print(f"\nOriginal y: {y}")

    y_norm = normalizer.normalize(y)
    print(f"Normalized y: {y_norm}")

    y_back = normalizer.denormalize(y_norm)
    print(f"Denormalized y: {y_back}")
    print(f"Reconstruction error: {(y - y_back).abs().max().item():.2e}")
