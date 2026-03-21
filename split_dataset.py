from typing import Tuple, Literal, Optional, List
import numpy as np
import pandas as pd

from dataset_loader import CircuitSeqDataset


SplitMode = Literal["by_circuit", "within_circuit"]


def _normalize_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[float, float, float]:
    s = float(train_ratio + val_ratio + test_ratio)
    if s <= 0:
        raise ValueError("Ratios must sum to a positive value.")
    return train_ratio / s, val_ratio / s, test_ratio / s


def split_dataset(
    ds: CircuitSeqDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    mode: SplitMode = "by_circuit",
    seed: int = 0,
    # within_circuit 可选：保证每个 gid 在 val/test 至少留 k 条（样本不够则尽力）
    min_per_split_per_gid: int = 0,
    # by_circuit 时：按电路 label 均值分层，使 train/val/test 桶分布相近
    stratify_labels: Optional[List[str]] = None,
) -> Tuple[CircuitSeqDataset, CircuitSeqDataset, CircuitSeqDataset]:
    """
    Split dataset AFTER optimization2 (i.e., ds.df has 'gid').

    Args:
      ds: CircuitSeqDataset with ds.df containing 'gid'
      train_ratio/val_ratio/test_ratio: split ratios (auto-normalized)
      mode:
        - "by_circuit": split by gid, circuits do NOT overlap across splits.
        - "within_circuit": split within each gid, every gid has train/val/test (if enough samples).
      seed: random seed
      min_per_split_per_gid: only for within_circuit
      stratify_labels: when mode=by_circuit, stratify circuits by per-circuit mean of first label,
        assign proportionally to train/val/test so bin distributions are similar across splits.

    Returns:
      train_ds, val_ds, test_ds (all share graph_pool via ds.make_subset)
    """
    if "gid" not in ds.df.columns:
        raise ValueError("ds.df must contain a 'gid' column. Please build dataset with optimization1/2 first.")

    train_ratio, val_ratio, test_ratio = _normalize_ratios(train_ratio, val_ratio, test_ratio)
    rng = np.random.default_rng(seed)

    if mode == "by_circuit":
        gids = np.array(sorted(ds.df["gid"].unique()))
        n_total = len(gids)

        if stratify_labels and len(stratify_labels) > 0:
            # Stratify circuits by per-circuit mean of first label
            first_label = stratify_labels[0]
            col = getattr(ds, "label2col", {}).get(first_label, first_label)
            if col not in ds.df.columns:
                raise ValueError(
                    f"stratify_labels[0]='{first_label}' (col='{col}') not in ds.df columns: {list(ds.df.columns)}"
                )
            per_gid_mean = ds.df.groupby("gid")[col].mean()
            gid_to_mean = per_gid_mean.to_dict()
            means = np.array([gid_to_mean[g] for g in gids])
            # Bin circuits by quantiles of per-circuit means (use ~min(10, n_total) bins)
            n_bins = min(10, max(2, n_total // 3))
            quantiles = np.percentile(means, np.linspace(0, 100, n_bins + 1)[1:-1])
            bin_indices = np.searchsorted(quantiles, means)
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            # Assign proportionally from each bin to train/val/test
            train_gids_list, val_gids_list, test_gids_list = [], [], []
            for b in range(n_bins):
                mask = bin_indices == b
                gids_in_bin = gids[mask]
                rng.shuffle(gids_in_bin)
                n_b = len(gids_in_bin)
                n_tr = int(round(n_b * train_ratio))
                n_va = int(round(n_b * val_ratio))
                n_tr = min(n_tr, n_b)
                n_va = min(n_va, n_b - n_tr)
                train_gids_list.extend(gids_in_bin[:n_tr].tolist())
                val_gids_list.extend(gids_in_bin[n_tr : n_tr + n_va].tolist())
                test_gids_list.extend(gids_in_bin[n_tr + n_va :].tolist())
            train_gids = set(train_gids_list)
            val_gids = set(val_gids_list)
            test_gids = set(test_gids_list)
        else:
            rng.shuffle(gids)
            n_train = int(round(n_total * train_ratio))
            n_val = int(round(n_total * val_ratio))
            n_train = min(n_train, n_total)
            n_val = min(n_val, n_total - n_train)
            train_gids = set(gids[:n_train].tolist())
            val_gids = set(gids[n_train : n_train + n_val].tolist())
            test_gids = set(gids[n_train + n_val :].tolist())

        df_train = ds.df[ds.df["gid"].isin(train_gids)].copy()
        df_val = ds.df[ds.df["gid"].isin(val_gids)].copy()
        df_test = ds.df[ds.df["gid"].isin(test_gids)].copy()

        return ds.make_subset(df_train), ds.make_subset(df_val), ds.make_subset(df_test)

    elif mode == "within_circuit":
        train_parts = []
        val_parts = []
        test_parts = []

        for gid, df_g in ds.df.groupby("gid", sort=True):
            df_g = df_g.reset_index(drop=True)
            idx = np.arange(len(df_g))
            rng.shuffle(idx)

            n = len(idx)
            n_train = int(round(n * train_ratio))
            n_val = int(round(n * val_ratio))

            # clamp
            n_train = min(n_train, n)
            n_val = min(n_val, n - n_train)

            # optional min-per-split enforcement
            if min_per_split_per_gid > 0 and n >= 3 * min_per_split_per_gid:
                n_train = max(n_train, min_per_split_per_gid)
                n_val = max(n_val, min_per_split_per_gid)
                # ensure test has at least min too
                n_train = min(n_train, n - 2 * min_per_split_per_gid)
                n_val = min(n_val, n - n_train - min_per_split_per_gid)

            n_test = n - n_train - n_val

            train_idx = idx[:n_train]
            val_idx = idx[n_train:n_train + n_val]
            test_idx = idx[n_train + n_val:]

            train_parts.append(df_g.iloc[train_idx])
            val_parts.append(df_g.iloc[val_idx])
            test_parts.append(df_g.iloc[test_idx])

        df_train = pd.concat(train_parts, axis=0).reset_index(drop=True)
        df_val = pd.concat(val_parts, axis=0).reset_index(drop=True)
        df_test = pd.concat(test_parts, axis=0).reset_index(drop=True)

        return ds.make_subset(df_train), ds.make_subset(df_val), ds.make_subset(df_test)

    else:
        raise ValueError(f"Unknown mode={mode}. Use 'by_circuit' or 'within_circuit'.")

