import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import dgl
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset_loader import CircuitSeqDataset, collate_circuit_seq
from split_dataset import split_dataset
from model import TopCircuitSeqModel, TopCircuitSeqModelCfg
from label_normalizer import LabelNormalizer, compute_metrics_original_space, compute_metrics_per_circuit
from quantile_bins import QuantileBinManager


# -----------------------------
# utils
# -----------------------------
def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _extract_pred(out):
    """model forward could return Tensor or (values, logits, ...)"""
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _extract_labels_from_batch(batch: Dict[str, Any], label_names: List[str], device: str) -> torch.Tensor:
    """
    从 batch 中提取独立 label，并拼成 (B, T) 的 tensor。
    batch 里应有 batch["area"], batch["delay"] 这种 (B,) 张量。
    """
    ys = []
    for lb in label_names:
        if lb not in batch:
            raise KeyError(f"Label '{lb}' not found in batch. Available keys: {list(batch.keys())}")
        y = batch[lb]
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        ys.append(y.to(device).to(torch.float32))  # (B,)
    return torch.stack(ys, dim=1)  # (B, T)


def mse_loss(pred_norm: torch.Tensor, y_norm: torch.Tensor) -> torch.Tensor:
    """
    pred_norm: (B,T) or (B,T,1)
    y_norm   : (B,T)
    """
    if pred_norm.ndim == 3 and pred_norm.shape[-1] == 1:
        pred_norm = pred_norm.squeeze(-1)
    if pred_norm.ndim != 2:
        raise ValueError(f"Unexpected pred shape: {tuple(pred_norm.shape)}")
    return ((pred_norm - y_norm) ** 2).mean()


def _split_batch_to_micro_batches(
    batch: Dict[str, Any], label_names: List[str], device: str
) -> List[Dict[str, Any]]:
    """
    将 batch 拆成多个 micro-batch（每个样本一个），用于 OOM 时梯度累积。
    """
    bg = batch["g"]
    seq = batch["seq"]
    seq_len = batch["seq_len"]
    B = seq.shape[0]

    graphs = dgl.unbatch(bg)
    micro_batches = []
    for i in range(B):
        mb = {
            "g": graphs[i],
            "seq": seq[i : i + 1].to(device),
            "seq_len": seq_len[i : i + 1].to(device),
        }
        for lb in label_names:
            mb[lb] = batch[lb][i : i + 1].to(device)
        if "gid" in batch:
            mb["gid"] = batch["gid"][i : i + 1].to(device)
        micro_batches.append(mb)
    return micro_batches


def _is_oom_error(e: Exception) -> bool:
    """判断是否为 CUDA OOM 错误"""
    oom_strs = ("out of memory", "CUDA out of memory", "cudaErrorOutOfMemory")
    err_str = str(e).lower()
    return any(s.lower() in err_str for s in oom_strs)


def combined_loss(
    values: torch.Tensor, 
    logits: Optional[torch.Tensor], 
    y_regression: torch.Tensor, 
    y_classification: Optional[torch.Tensor],
    num_classes: int,
    alpha: float = 1.0,
    task_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Combined loss: weighted MSE + Cross-Entropy Loss

    Args:
        values: (B, T, out_dim) - regression values
        logits: (B, T, C) - classification logits, None if classification is not used
        y_regression: (B, T) - 回归真值（用于 MSE）
        y_classification: (B, T) - 分类真值（分桶索引，用于 CE），如果为 None 则不计算 CE
        num_classes: Number of classes for classification
        alpha: 分类损失权重，默认为 1.0
        task_weights: (T,) 各任务的损失权重，None 时等权

    Returns:
        Total loss combining MSE and classification loss
    """
    values = values.squeeze(-1)  # (B, T)
    
    # Weighted MSE loss: per-task weighting
    per_sample_mse = (values - y_regression) ** 2  # (B, T)
    if task_weights is not None:
        w = task_weights.to(per_sample_mse.device)  # (T,)
        per_sample_mse = per_sample_mse * w.unsqueeze(0)  # (B, T) * (1, T)
    mse = per_sample_mse.mean()

    if logits is None or y_classification is None:
        return mse

    # Weighted CrossEntropy: per-task weighting
    T = logits.shape[1]
    ce_per_task = []
    for t in range(T):
        ce_t = nn.CrossEntropyLoss()(
            logits[:, t, :],  # (B, C)
            y_classification[:, t].long(),  # (B,)
        )
        ce_per_task.append(ce_t)
    ce_per_task = torch.stack(ce_per_task)  # (T,)
    if task_weights is not None:
        w = task_weights.to(ce_per_task.device)
        ce_loss = (ce_per_task * w).mean()
    else:
        ce_loss = ce_per_task.mean()

    total_loss = mse + alpha * ce_loss
    return total_loss


@torch.no_grad()
def compute_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    label_names: List[str],
    normalizer: Optional[LabelNormalizer] = None,
    eps: float = 1e-8,
    is_ddp: bool = False,
    use_oom_fallback: bool = True,
) -> Dict[str, Any]:
    """
    指标在 ORIGINAL space 计算：
      - 如果 normalizer != None：认为 model 输出的是“normalized space”的预测
        => 先 denormalize 再算 R²/MAPE/MAE/MSE
      - 否则：直接用 model 输出当 original
      - per_circuit: 按电路计算 R²，返回 r2_avg_over_circuits 作为训练目标
    """
    model.eval()
    preds = []
    trues = []
    gids_list = []

    for batch in loader:
        has_gid = "gid" in batch
        try:
            g = batch["g"].to(device)
            seq = batch["seq"].to(device)
            seq_len = batch["seq_len"].to(device)
            y = _extract_labels_from_batch(batch, label_names, device)
            out = model(g, seq, seq_len)
            p = _extract_pred(out)
        except RuntimeError as e:
            if use_oom_fallback and _is_oom_error(e):
                torch.cuda.empty_cache()
                micro_batches = _split_batch_to_micro_batches(batch, label_names, device)
                for mb in micro_batches:
                    g = mb["g"].to(device)
                    out = model(g, mb["seq"], mb["seq_len"])
                    p = _extract_pred(out)
                    if p.ndim == 3 and p.shape[-1] == 1:
                        p = p.squeeze(-1)
                    if normalizer is not None:
                        p = normalizer.denormalize(p)
                    y = _extract_labels_from_batch(mb, label_names, device)
                    preds.append(p.detach())
                    trues.append(y.detach())
                    if "gid" in mb:
                        gids_list.append(mb["gid"].detach())
                continue
            raise

        if p.ndim == 3 and p.shape[-1] == 1:
            p = p.squeeze(-1)
        if p.ndim != 2:
            raise ValueError(f"Unexpected pred shape: {tuple(p.shape)}")
        if normalizer is not None:
            p = normalizer.denormalize(p)
        preds.append(p.detach())
        trues.append(y.detach())
        if has_gid:
            gids_list.append(batch["gid"].to(device).detach())

    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)
    gids = torch.cat(gids_list, dim=0) if gids_list else None

    if is_ddp and dist.is_initialized() and dist.get_world_size() > 1:
        world_size = dist.get_world_size()
        local_size = pred.shape[0]
        size_tensor = torch.tensor([local_size], device=pred.device, dtype=torch.long)
        size_list = [torch.zeros_like(size_tensor) for _ in range(world_size)]
        dist.all_gather(size_list, size_tensor)
        max_size = max(s.item() for s in size_list)
        if local_size < max_size:
            pad_pred = torch.zeros(max_size - local_size, pred.shape[1], device=pred.device, dtype=pred.dtype)
            pad_true = torch.zeros(max_size - local_size, true.shape[1], device=true.device, dtype=true.dtype)
            pred = torch.cat([pred, pad_pred], dim=0)
            true = torch.cat([true, pad_true], dim=0)
            if gids is not None:
                pad_gid = torch.zeros(max_size - local_size, device=gids.device, dtype=gids.dtype)
                gids = torch.cat([gids, pad_gid], dim=0)
        pred_list = [torch.zeros_like(pred) for _ in range(world_size)]
        true_list = [torch.zeros_like(true) for _ in range(world_size)]
        dist.all_gather(pred_list, pred)
        dist.all_gather(true_list, true)
        if gids is not None:
            gid_list = [torch.zeros_like(gids) for _ in range(world_size)]
            dist.all_gather(gid_list, gids)
        preds_unpad, trues_unpad, gids_unpad = [], [], []
        for i, sz in enumerate(size_list):
            s = sz.item()
            preds_unpad.append(pred_list[i][:s])
            trues_unpad.append(true_list[i][:s])
            if gids is not None:
                gids_unpad.append(gid_list[i][:s])
        pred = torch.cat(preds_unpad, dim=0)
        true = torch.cat(trues_unpad, dim=0)
        if gids_unpad:
            gids = torch.cat(gids_unpad, dim=0)

    pred = pred.cpu()
    true = true.cpu()
    if gids is not None:
        gids = gids.cpu()

    if gids is not None:
        return compute_metrics_per_circuit(pred, true, gids, label_names, eps)
    return compute_metrics_original_space(pred, true, label_names, eps)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: str,
    grad_clip: float,
    label_names: List[str],
    num_classes: int,
    normalizer: Optional[LabelNormalizer] = None,
    bin_manager: Optional[QuantileBinManager] = None,
    ce_alpha: float = 1.0,
    task_weights: Optional[torch.Tensor] = None,
    use_oom_fallback: bool = True,
    is_ddp: bool = False,
    accum_steps: int = 1,
) -> float:
    """
    训练一个 epoch，使用 MSE 和分类损失之和。
    支持 OOM 时自动拆分为 micro-batch 梯度累积（动态 batch size）。
    支持 accum_steps 梯度累积以降低显存。
    
    Args:
        bin_manager: 分位数管理器，用于生成分类标签。如果为 None 则不使用分类损失。
        ce_alpha: 分类损失权重，默认为 1.0
        task_weights: (T,) 各任务的损失权重，None 时等权
    """
    model.train()
    total = 0.0
    n = 0
    accum_cnt = 0

    for batch in loader:
        bs = batch["seq"].shape[0]
        y = _extract_labels_from_batch(batch, label_names, device)  # (B,T)
        y_for_loss = normalizer.normalize(y) if normalizer is not None else y  # (B,T)
        
        # 生成分类标签（分桶索引）
        if bin_manager is not None and bin_manager.is_fitted:
            y_classification = bin_manager.get_bin_indices(y)  # (B, T)
        else:
            y_classification = None

        if accum_cnt == 0:
            opt.zero_grad(set_to_none=True)

        try:
            g = batch["g"].to(device)
            seq = batch["seq"].to(device)
            seq_len = batch["seq_len"].to(device)
            # 传入 target_bins 用于 teacher forcing
            out = model(g, seq, seq_len, target_bins=y_classification)
            values, logits = out
            loss = combined_loss(
                values, logits, y_for_loss, y_classification,
                num_classes=num_classes, alpha=ce_alpha,
                task_weights=task_weights,
            )
            loss = loss / accum_steps
            loss.backward()
            total += float(loss.detach().cpu()) * bs * accum_steps
            n += bs
            accum_cnt += 1
            if accum_cnt >= accum_steps:
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                accum_cnt = 0

        except RuntimeError as e:
            if use_oom_fallback and _is_oom_error(e):
                torch.cuda.empty_cache()
                if is_ddp:
                    if dist.get_rank() == 0:
                        print(f"  [OOM] batch_size={bs} -> fallback to micro-batches (size=1)", flush=True)
                else:
                    print(f"  [OOM] batch_size={bs} -> fallback to micro-batches (size=1)", flush=True)
                micro_batches = _split_batch_to_micro_batches(batch, label_names, device)
                opt.zero_grad(set_to_none=True)
                for mb in micro_batches:
                    mb_y = _extract_labels_from_batch(mb, label_names, device)
                    mb_y_for_loss = normalizer.normalize(mb_y) if normalizer is not None else mb_y
                    # 生成分类标签
                    if bin_manager is not None and bin_manager.is_fitted:
                        mb_y_classification = bin_manager.get_bin_indices(mb_y)
                    else:
                        mb_y_classification = None
                    g_mb = mb["g"].to(device)
                    out = model(g_mb, mb["seq"], mb["seq_len"], target_bins=mb_y_classification)
                    values, logits = out
                    loss = combined_loss(
                        values, logits, mb_y_for_loss, mb_y_classification,
                        num_classes=num_classes, alpha=ce_alpha,
                        task_weights=task_weights,
                    )
                    loss.backward()
                    total += float(loss.detach().cpu())
                    n += 1
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                accum_cnt = 0
            else:
                raise

    if accum_cnt > 0:
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

    return total / max(1, n)



# -----------------------------
# cfg
# -----------------------------
def setup_ddp(rank=None, world_size=None) -> tuple:
    """初始化 DDP，返回 (rank, local_rank, world_size, device).
    支持 torchrun 和 mp.spawn 两种启动方式。
    """
    # torchrun 方式：环境变量由 torchrun 自动设置
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
        return rank, local_rank, world_size, device
    # mp.spawn 方式：rank 和 world_size 由调用方传入
    if rank is not None and world_size is not None and world_size > 1:
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        device = f"cuda:{rank}"
        return rank, rank, world_size, device
    # 单卡/CPU 模式
    return 0, 0, 1, None


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


@dataclass
class TrainCfg:
    seed: int = 0
    split_mode: str = "within_circuit"

    batch_size: int = 2  # 大图显存紧张时可调小，配合 accum_steps 保持有效 batch
    accum_steps: int = 1  # 梯度累积步数，有效 batch = batch_size * accum_steps
    num_workers: int = 1
    use_ddp: bool = True  # 多卡时自动使用 DDP
    use_oom_fallback: bool = True  # OOM 时自动拆分为 micro-batch

    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 5000
    grad_clip: float = 1.0

    # early stopping: val average area R2 per circuit
    patience: int = 10
    r2_min_delta: float = 1e-6  # improvement threshold

    # label normalization
    use_label_normalization: bool = True
    
    # quantile bin manager (分位数分桶)
    use_quantile_bins: bool = True  # 是否使用分位数分桶
    num_bins: int = 8  # 分桶数量（分支数量）
    ce_alpha: float = 1.0  # 分类损失权重

    # per-task loss weights (与 task_names 对应，area 权重加大以提升准确性)
    task_loss_weights: tuple = (1.0, 3.0)  # (area, delay)

    # data paths
    csv_path: str = "/home/yfdai/asap/data/output_large.csv"
    circuit_dir: str = "/home/yfdai/asap/data/aag"
    seq_dir: str = "/home/yfdai/asap/data/seq_large_new/"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./ckpt"
    save_name: str = "ad_fpl26_aag.pt"
    normalizer_name: str = "label_normalizer_adr_fpl26.json"
    bin_manager_name: str = "quantile_adr_bins.json"


def build_dataset(
    labels: List[str],
    csv_path: str,
    circuit_dir: str,
    seq_dir: str,
    verbose: bool = True,
) -> CircuitSeqDataset:
    """
    只加载你关心的 labels（area/delay）。
    verbose=False 时跳过 preload 日志（DDP 下仅 rank 0 打印）。
    """
    kwargs: Dict[str, Any] = dict(
        csv_path=csv_path,
        circuit_dir=circuit_dir,
        seq_dir=seq_dir,
        use_header=True,
        check_paths=False,
        preload_graphs=True,
        labels=labels,
    )
    # 兼容旧版 dataset_loader（无 verbose 参数）
    import inspect
    if "verbose" in inspect.signature(CircuitSeqDataset.__init__).parameters:
        kwargs["verbose"] = verbose
    ds = CircuitSeqDataset(**kwargs)
    return ds


def main(rank=None, world_size=None):
    cfg = TrainCfg()
    set_seed(cfg.seed)

    # 缓解 CUDA 显存碎片，可在环境变量中设置 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ and torch.cuda.is_available():
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    # -----------------------------
    # DDP 初始化（支持 torchrun 和 mp.spawn 两种方式）
    # -----------------------------
    rank, local_rank, world_size, ddp_device = setup_ddp(rank, world_size)
    is_ddp = world_size > 1
    device = ddp_device if is_ddp else (cfg.device if torch.cuda.is_available() else "cpu")

    if is_ddp and rank == 0:
        print(f"[DDP] world_size={world_size}, using NCCL backend", flush=True)

    # -----------------------------
    # Only two tasks
    # -----------------------------
    task_names = ["area", "delay"]
    num_tasks = len(task_names)

    # -----------------------------
    # 动态生成文件名（包含 csv 名称和预测指标）
    # -----------------------------
    csv_basename = os.path.splitext(os.path.basename(cfg.csv_path))[0]  # 去掉路径和 .csv
    tasks_str = "_".join(task_names)  # e.g., "area_delay"
    name_suffix = f"{csv_basename}_{tasks_str}"  # e.g., "vtr_abcd_epfl_iscas_area_delay"
    
    save_name = f"best_r2_within_circuit_{name_suffix}.pt"
    normalizer_name = f"label_normalizer_{name_suffix}.json"
    bin_manager_name = f"quantile_bins_{name_suffix}.json"

    # -----------------------------
    # Dataset + Split (within_circuit)
    # -----------------------------
    ds = build_dataset(
        labels=task_names,
        csv_path=cfg.csv_path,
        circuit_dir=cfg.circuit_dir,
        seq_dir=cfg.seq_dir,
        verbose=(rank == 0),
    )
    if "gid" not in ds.df.columns:
        raise ValueError("ds.df must contain a 'gid' column to use split_dataset().")

    # -----------------------------
    # Quantile Bin Manager (fit on FULL dataset BEFORE split)
    # -----------------------------
    bin_manager: Optional[QuantileBinManager] = None
    if cfg.use_quantile_bins:
        if rank == 0:
            print("\n[INFO] Computing quantile bin boundaries from FULL dataset (ds.df) ...", flush=True)

        # Extract labels from full ds.df
        full_labels = []
        for lb in task_names:
            col = getattr(ds, "label2col", {}).get(lb, lb)
            if col not in ds.df.columns:
                raise KeyError(f"Label '{lb}' (col='{col}') not in ds.df columns: {list(ds.df.columns)}")
            full_labels.append(ds.df[col].astype(float).values)
        import numpy as np
        full_labels = np.stack(full_labels, axis=1)  # (N, T)

        bin_manager = QuantileBinManager(num_tasks=num_tasks, num_bins=cfg.num_bins)
        bin_manager.fit(full_labels)

        if rank == 0:
            bin_manager_path = os.path.join(cfg.save_dir, bin_manager_name)
            os.makedirs(cfg.save_dir, exist_ok=True)
            bin_manager.save(bin_manager_path)
            print(f"[INFO] Quantile bin manager fitted on full dataset, saved to {bin_manager_path}")
            # 打印完整数据集的桶边界和分布
            import numpy as np
            full_stats = bin_manager.get_bin_statistics(full_labels)
            print(f"[INFO] Full dataset bin distribution (n={len(full_labels)}, num_bins={cfg.num_bins}):")
            for t, name in enumerate(task_names):
                ts = full_stats[f"task_{t}"]
                bnd = ts.get("boundaries", [])
                print(f"  {name}: boundaries={[f'{b:.4f}' for b in bnd]}")
                for b in range(cfg.num_bins):
                    bi = ts.get(f"bin_{b}", {})
                    cnt, pct = bi.get("count", 0), bi.get("percentage", 0)
                    print(f"    bin_{b}: count={cnt} ({pct:.1f}%)")

        if is_ddp:
            dist.barrier()
        if is_ddp and rank != 0:
            bin_manager_path = os.path.join(cfg.save_dir, bin_manager_name)
            bin_manager = QuantileBinManager.load(bin_manager_path)

    train_ds, val_ds, test_ds = split_dataset(
        ds,
        train_ratio=0.6,
        val_ratio=0.1,
        test_ratio=0.3,
        mode="within_circuit",
        seed=cfg.seed,
        min_per_split_per_gid=0,
        stratify_labels=task_names,
    )

    # -----------------------------
    # Label Normalizer (fit on train only, rank 0 only for DDP)
    # -----------------------------
    normalizer: Optional[LabelNormalizer] = None
    if rank == 0:
        os.makedirs(cfg.save_dir, exist_ok=True)

    if cfg.use_label_normalization:
        if rank == 0:
            print("\n[INFO] Computing label statistics from train_ds ONLY ...", flush=True)
        normalizer = LabelNormalizer(labels=task_names)
        normalizer.compute_stats(train_ds)
        if rank == 0:
            normalizer.print_stats()
            normalizer_path = os.path.join(cfg.save_dir, normalizer_name)
            normalizer.save(normalizer_path)
        if is_ddp:
            dist.barrier()
        if is_ddp and rank != 0:
            normalizer_path = os.path.join(cfg.save_dir, normalizer_name)
            normalizer = LabelNormalizer.load(normalizer_path)
    if not cfg.use_label_normalization and rank == 0:
        print("\n[INFO] Label normalization is DISABLED.", flush=True)

    # -----------------------------
    # DataLoaders（DDP 使用 DistributedSampler）
    # -----------------------------
    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if is_ddp
        else None
    )
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if is_ddp
        else None
    )
    test_sampler = (
        DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if is_ddp
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_circuit_seq,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_circuit_seq,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_circuit_seq,
    )

    # -----------------------------
    # Print bin distribution for train, val, test splits
    # -----------------------------
    if cfg.use_quantile_bins and bin_manager is not None and rank == 0:
        import numpy as np

        def _extract_labels_from_ds(dataset) -> np.ndarray:
            labels = []
            for i in range(len(dataset)):
                sample = dataset[i]
                label_row = [sample.y[lb].item() if torch.is_tensor(sample.y[lb]) else float(sample.y[lb]) for lb in task_names]
                labels.append(label_row)
            return np.array(labels)

        def _print_bin_stats(split_name: str, labels_arr: np.ndarray, n_circuits: int = 0):
            if len(labels_arr) == 0:
                print(f"  [{split_name}] (empty)")
                return
            stats = bin_manager.get_bin_statistics(labels_arr)
            circ_str = f", n_circuits={n_circuits}" if n_circuits > 0 else ""
            print(f"  [{split_name}] n_samples={len(labels_arr)}{circ_str}")
            for t, name in enumerate(task_names):
                task_stats = stats[f"task_{t}"]
                boundaries = task_stats.get("boundaries", [])
                print(f"    {name}: boundaries={[f'{b:.4f}' for b in boundaries]}")
                for b in range(cfg.num_bins):
                    bin_info = task_stats.get(f"bin_{b}", {})
                    count = bin_info.get("count", 0)
                    pct = bin_info.get("percentage", 0)
                    print(f"      bin_{b}: count={count} ({pct:.1f}%)")

        def _n_circuits(d):
            return len(d.df["gid"].unique()) if "gid" in d.df.columns else 0

        print(f"\n[INFO] Bin distribution across splits (num_bins={cfg.num_bins}, stratify by circuit):")
        _print_bin_stats("TRAIN", _extract_labels_from_ds(train_ds), _n_circuits(train_ds))
        _print_bin_stats("VAL", _extract_labels_from_ds(val_ds), _n_circuits(val_ds))
        _print_bin_stats("TEST", _extract_labels_from_ds(test_ds), _n_circuits(test_ds))

    if not cfg.use_quantile_bins and rank == 0:
        print("\n[INFO] Quantile bin classification is DISABLED.", flush=True)

    # -----------------------------
    # Model
    # -----------------------------
    # 当使用分位数分桶时，ens_num_classes 使用 num_bins；否则为 1（单分支）
    ens_num_classes = cfg.num_bins if cfg.use_quantile_bins else 1
    
    model_cfg = TopCircuitSeqModelCfg(
        num_tasks=num_tasks,
        gin_in_dim=8,
        gin_hidden_dim=128,
        gin_layers=2,
        seq_in_dim=31,
        seq_hidden_dim=128,
        seq_layers=2,
        fe_num_heads=4,
        fe_num_layers=2,
        fs_num_heads=4,
        fs_num_layers=2,
        ens_num_classes=ens_num_classes,
        ens_num_layers=3,
        ens_hidden_dim=256,
    )
    model = TopCircuitSeqModel(model_cfg).to(device)
    if is_ddp and cfg.use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # -----------------------------
    # Early stopping
    # -----------------------------
    best_score = -1e18
    best_epoch = -1
    bad_count = 0
    best_path = os.path.join(cfg.save_dir, save_name)

    if rank == 0:
        print("\n" + "=" * 70)
        print(f"[INFO] split_mode=within_circuit")
        print(f"[INFO] tasks={task_names}")
        print(f"[INFO] max_epochs={cfg.max_epochs}, early_stop_patience={cfg.patience} (val avg area R2 per circuit)")
        print(f"[INFO] use_label_normalization={cfg.use_label_normalization}")
        print(f"[INFO] use_quantile_bins={cfg.use_quantile_bins}, num_bins={cfg.num_bins}, ce_alpha={cfg.ce_alpha}")
        print(f"[INFO] DDP={is_ddp} world_size={world_size}")
        print(f"[INFO] batch_size={cfg.batch_size} accum_steps={cfg.accum_steps} (eff_batch={cfg.batch_size * cfg.accum_steps})")
        print(f"[INFO] use_oom_fallback={cfg.use_oom_fallback} (dynamic batch size)")
        print(f"[INFO] task_loss_weights={dict(zip(task_names, cfg.task_loss_weights))}")
        print("=" * 70 + "\n")

    task_weights = torch.tensor(list(cfg.task_loss_weights), dtype=torch.float32)

    for epoch in range(1, cfg.max_epochs + 1):
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        tr_loss = train_one_epoch(
            model,
            train_loader,
            opt,
            device,
            cfg.grad_clip,
            num_classes=model_cfg.ens_num_classes,
            label_names=task_names,
            normalizer=normalizer,
            bin_manager=bin_manager,
            ce_alpha=cfg.ce_alpha,
            task_weights=task_weights,
            use_oom_fallback=cfg.use_oom_fallback,
            is_ddp=is_ddp,
            accum_steps=cfg.accum_steps,
        )

        tr = compute_metrics(
            model, train_loader, device, task_names, normalizer=normalizer,
            is_ddp=is_ddp, use_oom_fallback=cfg.use_oom_fallback,
        )
        va = compute_metrics(
            model, val_loader, device, task_names, normalizer=normalizer,
            is_ddp=is_ddp, use_oom_fallback=cfg.use_oom_fallback,
        )
        # 保存策略：当 r2_per_circuit 中 area 的 R² 在所有电路上的平均值最高时保存
        def _avg_area_r2_from_per_circuit(m: Dict[str, Any]) -> float:
            """从 r2_per_circuit 提取每个电路的 area R²，返回这些 area R² 的平均值（用于 best checkpoint 保存）"""
            if "r2_per_circuit" not in m:
                return m.get("r2_avg_over_circuits", m["r2_mean"])
            area_idx = task_names.index("area") if "area" in task_names else 0
            r2_per = m["r2_per_circuit"]
            if not r2_per:
                return m.get("r2_avg_over_circuits", m["r2_mean"])
            area_r2s = [r2_list[area_idx] for r2_list in r2_per.values() if len(r2_list) > area_idx]
            return sum(area_r2s) / len(area_r2s) if area_r2s else 0.0

        tr_r2 = _avg_area_r2_from_per_circuit(tr)
        va_r2 = _avg_area_r2_from_per_circuit(va)
        score_r2 = va_r2

        def fmt(m: Dict[str, Any]) -> str:
            r2s = " ".join([f"{task_names[i]}={m['r2_per_task'][i]:.4f}" for i in range(num_tasks)])
            mapes = " ".join([f"{task_names[i]}={m['mape_per_task'][i]*100:.2f}%" for i in range(num_tasks)])
            r2_val = m.get("r2_avg_over_circuits", m["r2_mean"])
            mape_val = m.get("mape_avg_over_circuits", m["mape_mean"])
            return (
                f"mse={m['mse']:.6f} R2_avg_within_circuit={r2_val:.4f} R2_mean={m['r2_mean']:.4f} | {r2s} "
                f"MAPE_avg_within_circuit={mape_val*100:.2f}% MAPE_mean={m['mape_mean']*100:.2f}% | {mapes}"
            )

        def print_per_circuit(m: Dict[str, Any], split_name: str, ds_ref):
            """打印每个电路的 R² 和 MAPE（均为电路内部计算）"""
            if "r2_per_circuit" not in m:
                return
            r2_per = m["r2_per_circuit"]
            r2_mean_per = m.get("r2_mean_per_circuit", {})
            mape_per = m.get("mape_per_circuit", {})
            mape_mean_per = m.get("mape_mean_per_circuit", {})
            for gid in sorted(r2_per.keys()):
                r2_list = r2_per[gid]
                r2_mean = r2_mean_per.get(gid, sum(r2_list) / len(r2_list))
                mape_list = mape_per.get(gid, [0.0] * len(r2_list))
                mape_mean = mape_mean_per.get(gid, sum(mape_list) / len(mape_list) if mape_list else 0.0)
                task_r2s = " ".join([f"{task_names[i]}={r2_list[i]:.4f}" for i in range(len(r2_list))])
                task_mapes = " ".join([f"{task_names[i]}={mape_list[i]*100:.2f}%" for i in range(len(mape_list))])
                circuit_name = os.path.basename(str(ds_ref.circuits[gid])) if ds_ref and gid < len(ds_ref.circuits) else f"gid{gid}"
                print(f"    [{split_name}] gid={gid:>3d} circuit={circuit_name} R2_avg={r2_mean:.4f} MAPE_avg={mape_mean*100:.2f}% | R2: {task_r2s} | MAPE: {task_mapes}")

        if rank == 0:
            print(f"[Epoch {epoch:04d}] train_loss(norm)={tr_loss:.6f}  val_avg_area_R2_per_circuit={score_r2:.6f}")
            print(f"  TRAIN {fmt(tr)}")
            print_per_circuit(tr, "TRAIN", train_ds)
            print(f"  VAL   {fmt(va)}")
            print_per_circuit(va, "VAL", val_ds)

        if score_r2 > best_score + cfg.r2_min_delta:
            best_score = score_r2
            best_epoch = epoch
            bad_count = 0

            if rank == 0:
                state_dict = model.module.state_dict() if is_ddp else model.state_dict()
                opt_state = opt.state_dict()
                torch.save(
                    {
                        "model": state_dict,
                        "opt": opt_state,
                        "epoch": epoch,
                        "best_score_avg_area_r2": best_score,
                        "task_names": task_names,
                        "csv_path": cfg.csv_path,
                        "csv_name": os.path.basename(cfg.csv_path),
                        "train_metrics": {
                            "r2_mean": tr["r2_mean"],
                            "r2_avg_over_circuits": tr.get("r2_avg_over_circuits", tr["r2_mean"]),
                            "r2_area_avg_over_circuits": tr_r2,
                            "r2_per_task": {name: tr["r2_per_task"][i] for i, name in enumerate(task_names)},
                            "mape_mean": tr["mape_mean"],
                            "mape_per_task": {name: tr["mape_per_task"][i] for i, name in enumerate(task_names)},
                            "mse": tr["mse"],
                        },
                        "val_metrics": {
                            "r2_mean": va["r2_mean"],
                            "r2_avg_over_circuits": va.get("r2_avg_over_circuits", va["r2_mean"]),
                            "r2_area_avg_over_circuits": va_r2,
                            "r2_per_task": {name: va["r2_per_task"][i] for i, name in enumerate(task_names)},
                            "mape_mean": va["mape_mean"],
                            "mape_per_task": {name: va["mape_per_task"][i] for i, name in enumerate(task_names)},
                            "mse": va["mse"],
                        },
                        "use_normalization": cfg.use_label_normalization,
                        "use_quantile_bins": cfg.use_quantile_bins,
                        "num_bins": cfg.num_bins,
                        "bin_manager_path": os.path.join(cfg.save_dir, bin_manager_name) if cfg.use_quantile_bins else None,
                    },
                    best_path,
                )
                print(f"  -> save best: {best_path} (best_val_avg_area_R2={best_score:.6f})")
        else:
            bad_count += 1
            if bad_count >= cfg.patience:
                if rank == 0:
                    print(f"\n[EARLY STOP] No val avg area R2 improvement for {cfg.patience} epochs.")
                    print(f"Best epoch={best_epoch}, best_val_avg_area_R2={best_score:.6f}")
                break

    # -----------------------------
    # Load best and evaluate on test
    # -----------------------------
    if rank == 0:
        print("\n" + "=" * 70)
        print("[FINAL] Loading best checkpoint and evaluating on TEST ...")
    ckpt = torch.load(best_path, map_location=device)
    load_model = model.module if is_ddp else model
    load_model.load_state_dict(ckpt["model"])

    te = compute_metrics(
        model, test_loader, device, task_names, normalizer=normalizer,
        is_ddp=is_ddp, use_oom_fallback=cfg.use_oom_fallback,
    )

    if rank == 0:
        te_r2 = te.get("r2_avg_over_circuits", te["r2_mean"])
        te_mape = te.get("mape_avg_over_circuits", te["mape_mean"])
        print(f"[BEST] epoch={ckpt['epoch']}  best_val_avg_area_R2_per_circuit={ckpt['best_score_avg_area_r2']:.6f}")
        print(f"[TEST] mse={te['mse']:.6f}  R2_avg_within_circuit={te_r2:.4f}  R2_mean={te['r2_mean']:.4f}  MAPE_avg_within_circuit={te_mape*100:.2f}%  MAPE_mean={te['mape_mean']*100:.2f}%")
        for i, name in enumerate(task_names):
            print(f"  {name}: R2={te['r2_per_task'][i]:.4f}  MAPE={te['mape_per_task'][i]*100:.2f}%  MAE={te['mae_per_task'][i]:.6f}")
        if "r2_per_circuit" in te:
            print("  [TEST] Per-circuit R2 & MAPE (circuit-internal):")
            for gid in sorted(te["r2_per_circuit"].keys()):
                r2_list = te["r2_per_circuit"][gid]
                r2_mean = te.get("r2_mean_per_circuit", {}).get(gid, sum(r2_list) / len(r2_list))
                mape_list = te.get("mape_per_circuit", {}).get(gid, [0.0] * len(r2_list))
                mape_mean = te.get("mape_mean_per_circuit", {}).get(gid, sum(mape_list) / len(mape_list) if mape_list else 0.0)
                task_r2s = " ".join([f"{task_names[i]}={r2_list[i]:.4f}" for i in range(len(r2_list))])
                task_mapes = " ".join([f"{task_names[i]}={mape_list[i]*100:.2f}%" for i in range(len(mape_list))])
                circuit_name = os.path.basename(str(test_ds.circuits[gid])) if gid < len(test_ds.circuits) else f"gid{gid}"
                print(f"    gid={gid:>3d} circuit={circuit_name} R2_avg={r2_mean:.4f} MAPE_avg={mape_mean*100:.2f}% | R2: {task_r2s} | MAPE: {task_mapes}")
        print("=" * 70)

    cleanup_ddp()


if __name__ == "__main__":
    import torch.multiprocessing as mp

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"

    # 方式一：如果由 torchrun 启动，环境变量已设好，直接调用 main()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        main()
    else:
        # 方式二：直接 python train.py，自动用 mp.spawn 启动多卡
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            if "MASTER_PORT" not in os.environ:
                import socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", 0))
                    os.environ["MASTER_PORT"] = str(s.getsockname()[1])
            print(f"[INFO] 检测到 {num_gpus} 张 GPU，通过 mp.spawn 启动多卡并行训练 (port={os.environ['MASTER_PORT']}) ...")
            mp.spawn(main, args=(num_gpus,), nprocs=num_gpus, join=True)
        else:
            print("[INFO] 仅检测到 1 张 GPU（或无 GPU），以单卡模式运行")
            main()
