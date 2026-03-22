"""
扫描 log 目录下的 *.log，解析电路名；对每电路加载对应 ckpt（iwls26_<circuit>.pt），
在 test 集（与 train.py 相同的 within_circuit 划分）上做 CPU 推理；
用 area×delay 作为标量指标：按「预测乘积最大」的一侧做切分比例（50/75/80%），
与「真值乘积最小」的 10%、5% 集合的误伤率写入 CSV：交集占该真值最小集合大小的百分比（0–100，与 train.py 中 mis_hit_rate 一致）。

依赖：与 train.py / infer_per_circuit.py 相同的数据路径与划分逻辑。
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from aig_preprocess_seq import aag_to_dgl_graph
from dataset_loader import pad_sequences
from infer_per_circuit import _build_model, _extract_pred, _forward_with_cached_g_emb
from label_normalizer import LabelNormalizer
from model import load_pt_checkpoint
from seq_preprocessing import load_seq
from split_dataset import split_dataset
from train import build_dataset, pruning_overlap_top_pred_largest_vs_true_smallest


def _parse_circuit_from_log(log_path: Path) -> str:
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline()
    m = re.search(r"========\s+\S+\s+(\S+)\s+->", first)
    if m:
        return m.group(1).strip()
    return log_path.stem


def _resolve_normalizer_path(ckpt: Dict[str, Any], ckpt_path: str, save_dir: str) -> Optional[str]:
    p = ckpt.get("normalizer_path")
    if isinstance(p, str) and os.path.isfile(p):
        return p
    flt = ""
    stem = os.path.basename(ckpt_path)
    if stem.startswith("iwls26_") and stem.endswith(".pt"):
        flt = stem[len("iwls26_") : -3]
    if flt:
        cand = os.path.join(save_dir, f"iwls26_{flt}_normalizer.json")
        if os.path.isfile(cand):
            return cand
    return None


@torch.no_grad()
def _infer_test_product(
    test_ds: Any,
    model: torch.nn.Module,
    task_names: List[str],
    device: torch.device,
    normalizer: Optional[LabelNormalizer],
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 test 集上逐样本的 (pred_area*delay, true_area*delay)，顺序与 test_ds 行顺序一致。"""
    all_gids = sorted(test_ds.df["gid"].unique().tolist())
    gid2indices: Dict[int, List[int]] = {g: test_ds.df[test_ds.df["gid"] == g].index.tolist() for g in all_gids}

    pred_parts: List[torch.Tensor] = []
    true_parts: List[torch.Tensor] = []

    for gid in all_gids:
        circuit_path = test_ds.circuits[gid]
        indices = gid2indices[gid]
        n_samples = len(indices)

        g_circuit, _ = aag_to_dgl_graph(circuit_path)
        g_dev = g_circuit.to(device)
        h0 = g_dev.ndata["nf"].to(torch.float32)
        node_emb = model.gin(g_dev, h0)
        g_emb = model._graph_pool(g_dev, node_emb)
        del g_dev, h0, node_emb, g_circuit

        for batch_start in range(0, n_samples, batch_size):
            batch_idx = indices[batch_start : batch_start + batch_size]
            seqs: List[torch.Tensor] = []
            true_rows: List[List[float]] = []
            for idx in batch_idx:
                row = test_ds.df.iloc[idx]
                seq_raw = load_seq(str(row[test_ds.seq_col]))
                if not torch.is_tensor(seq_raw):
                    seq_raw = torch.tensor(seq_raw)
                seqs.append(seq_raw.to(torch.float32))
                true_rows.append([float(row[test_ds.label2col[lb]]) for lb in task_names])

            seq_pad, seq_len = pad_sequences(seqs)
            seq_pad = seq_pad.to(device)
            seq_len = seq_len.to(device)

            out = _forward_with_cached_g_emb(model, g_emb, seq_pad, seq_len)
            pred = _extract_pred(out)
            if normalizer is not None:
                pred = normalizer.denormalize(pred)

            true = torch.tensor(true_rows, dtype=torch.float32)
            pred_parts.append(pred.detach().cpu())
            true_parts.append(true)

        del g_emb

    pred_t = torch.cat(pred_parts, dim=0)
    true_t = torch.cat(true_parts, dim=0)
    prod_p = (pred_t[:, 0] * pred_t[:, 1]).numpy()
    prod_t = (true_t[:, 0] * true_t[:, 1]).numpy()
    return prod_p, prod_t


def run(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    task_names = ["area", "delay"]
    log_dir = Path(args.log_dir)
    ckpt_dir = args.ckpt_dir
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    log_files = sorted(log_dir.glob("*.log"))
    if not log_files:
        raise FileNotFoundError(f"未在 {log_dir} 找到 *.log")

    cut_pcts = tuple(int(x) for x in args.cut_pred_largest_pcts.split(","))
    true_pcts = tuple(int(x) for x in args.true_smallest_pcts.split(","))

    rows: List[List[Any]] = []
    n_metric_cols = len(cut_pcts) * len(true_pcts)
    for log_path in log_files:
        circuit = _parse_circuit_from_log(log_path)
        ckpt_path = os.path.join(ckpt_dir, f"iwls26_{circuit}.pt")
        row_head = [circuit, str(log_path)]

        if not os.path.isfile(ckpt_path):
            rows.append(row_head + ["missing_ckpt", ""] + [""] * n_metric_cols)
            print(f"[SKIP] {circuit}: 无检查点 {ckpt_path}")
            continue

        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)

        num_bins = int(ckpt.get("num_bins", args.num_bins))
        model = _build_model(task_names, num_bins).to(device)
        load_pt_checkpoint(model, ckpt_path, map_location=device, strict=args.strict, ckpt=ckpt)
        model.eval()

        norm_path = _resolve_normalizer_path(ckpt, ckpt_path, ckpt_dir)
        normalizer = LabelNormalizer.load(norm_path) if norm_path else None

        ds = build_dataset(
            labels=task_names,
            csv_path=args.csv_path,
            circuit_dir=args.circuit_dir,
            seq_dir=args.seq_dir,
            verbose=False,
            circuit_name_filter=circuit,
        )
        _, _, test_ds = split_dataset(
            ds,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            mode="within_circuit",
            seed=args.seed,
            min_per_split_per_gid=0,
        )
        n_test = len(test_ds)
        if n_test == 0:
            rows.append(row_head + ["empty_test", 0] + [""] * n_metric_cols)
            print(f"[SKIP] {circuit}: test 集为空")
            continue

        prod_pred, prod_true = _infer_test_product(
            test_ds, model, task_names, device, normalizer, args.batch_size
        )

        metrics: List[Any] = []
        for cp in cut_pcts:
            for tq in true_pcts:
                st = pruning_overlap_top_pred_largest_vs_true_smallest(
                    prod_pred, prod_true, cp / 100.0, tq / 100.0
                )
                # 占「真值最小 q%」集合大小的比例 → 百分比（误伤率）
                metrics.append(round(float(st["mis_hit_rate"]) * 100.0, 6))

        rows.append(row_head + ["ok", n_test] + metrics)
        print(f"[OK] {circuit} n_test={n_test} pct_of_truemin_subset={metrics}")

    headers = ["circuit", "log_path", "status", "n_test"]
    for cp in cut_pcts:
        for tq in true_pcts:
            headers.append(f"hit_pct_of_truemin{tq}_subset_predmax{cp}")

    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

    print(f"\n[OK] 已写入: {args.output_csv}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="按 log 批量推理 test 集，输出 area×delay 剪枝交集 CSV（CPU）。")
    p.add_argument("--log-dir", type=str, default="./log")
    p.add_argument("--ckpt-dir", type=str, default="./ckpt")
    p.add_argument("--csv-path", type=str, default="/home/yfdai/asap/data/output_large.csv")
    p.add_argument("--circuit-dir", type=str, default="/home/yfdai/asap/data/aag")
    p.add_argument("--seq-dir", type=str, default="/home/yfdai/asap/data/seq_large_new/")
    p.add_argument("--output-csv", type=str, default="./pruning_product_intersections.csv")
    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-bins", type=int, default=8, help="ckpt 无 num_bins 元数据时的回退值")
    p.add_argument(
        "--cut-pred-largest-pcts",
        type=str,
        default="50,75,80",
        help="按预测 area×delay 从大到小取的一侧占比（百分数），逗号分隔",
    )
    p.add_argument(
        "--true-smallest-pcts",
        type=str,
        default="10,5",
        help="真值 area×delay 最小的一侧占比（百分数），逗号分隔；与 cut 顺序组合成列",
    )
    p.add_argument("--strict", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
