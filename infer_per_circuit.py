import argparse
import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from aig_preprocess_seq import aag_to_dgl_graph
from dataset_loader import CircuitSeqDataset, pad_sequences
from label_normalizer import LabelNormalizer, compute_metrics_original_space
from model import TopCircuitSeqModel, TopCircuitSeqModelCfg, load_pt_checkpoint
from seq_preprocessing import load_seq
from split_dataset import split_dataset


def _extract_pred(out: Any) -> torch.Tensor:
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.ndim == 3 and out.shape[-1] == 1:
        out = out.squeeze(-1)
    if out.ndim != 2:
        raise ValueError(f"Unexpected prediction shape: {tuple(out.shape)}")
    return out


def _build_model(task_names: List[str], num_bins: int) -> TopCircuitSeqModel:
    cfg = TopCircuitSeqModelCfg(
        num_tasks=len(task_names),
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
        ens_num_classes=max(1, int(num_bins)),
        ens_num_layers=3,
        ens_hidden_dim=256,
    )
    return TopCircuitSeqModel(cfg)


def _guess_normalizer_path(ckpt_path: str) -> Optional[str]:
    ckpt_name = os.path.basename(ckpt_path)
    stem = ckpt_name[:-3] if ckpt_name.endswith(".pt") else ckpt_name
    candidates = []
    if stem.startswith("best_r2_"):
        suffix = stem[len("best_r2_") :]
        candidates.append(f"label_normalizer_{suffix}.json")
    candidates.append("label_normalizer_vtr_abcd_epfl_iscas_area_delay.json")

    ckpt_dir = os.path.dirname(ckpt_path) or "."
    for name in candidates:
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            return p
    return None


def _forward_with_cached_g_emb(
    model: TopCircuitSeqModel,
    g_emb: torch.Tensor,
    seq: torch.Tensor,
    seq_len: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the model's downstream layers using a pre-computed graph embedding.

    This avoids dgl.batch and re-encoding the same graph for every batch,
    which is the main source of OOM when the circuit graph is large.
    """
    s_emb = model.lstm(seq, seq_len)
    bsz = s_emb.shape[0]
    base = torch.cat([g_emb.expand(bsz, -1), s_emb], dim=-1)
    base = model.fuse(base)

    T = model.T
    D = base.shape[-1]
    F_task = base.unsqueeze(1).expand(bsz, T, D).contiguous()

    F_out = []
    for i in range(T):
        F_out.append(model.feature_extractors[i](F_task[:, i, :]))
    F_task = torch.stack(F_out, dim=1)

    F_task = model.feature_sharing(F_task)

    values = []
    logits_list = []
    has_logits = False
    for i in range(T):
        yi, li = model.ensembles[i](F_task[:, i, :], target_bin=None)
        values.append(yi)
        if li is not None:
            logits_list.append(li)
            has_logits = True
    values = torch.stack(values, dim=1)
    logits = torch.stack(logits_list, dim=1) if has_logits and len(logits_list) == T else None
    return values, logits


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> None:
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
    task_names = [x.strip() for x in args.tasks.split(",") if x.strip()]
    if len(task_names) == 0:
        raise ValueError("`--tasks` cannot be empty.")

    ds = CircuitSeqDataset(
        csv_path=args.csv_path,
        circuit_dir=args.circuit_dir,
        seq_dir=args.seq_dir,
        use_header=True,
        check_paths=False,
        preload_graphs=False,
        labels=task_names,
    )

    if args.split in ("by_circuit", "within_circuit"):
        _, _, ds_eval = split_dataset(
            ds,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            mode=args.split,
            seed=args.seed,
            min_per_split_per_gid=0,
        )
    else:
        ds_eval = ds

    model = _build_model(task_names, args.num_bins).to(device)
    ckpt_meta = load_pt_checkpoint(model, args.ckpt_path, map_location=device, strict=args.strict)
    model.eval()

    normalizer = None
    normalizer_path = args.normalizer_path
    if not normalizer_path:
        normalizer_path = _guess_normalizer_path(args.ckpt_path)
    if normalizer_path and os.path.exists(normalizer_path):
        normalizer = LabelNormalizer.load(normalizer_path)
        print(f"[INFO] use normalizer: {normalizer_path}")
    else:
        print("[INFO] no normalizer loaded; prediction is treated as original space.")

    all_gids = sorted(ds_eval.df["gid"].unique().tolist())
    gid2indices: Dict[int, List[int]] = {}
    for gid in all_gids:
        gid2indices[gid] = ds_eval.df[ds_eval.df["gid"] == gid].index.tolist()

    rows: List[List[Any]] = []
    print(f"\n[INFO] Total circuits: {len(all_gids)}, total samples: {len(ds_eval)}")
    print("===== Per-circuit inference (one circuit at a time) =====")

    for ci, gid in enumerate(all_gids):
        circuit_path = ds_eval.circuits[gid]
        circuit_name = os.path.basename(str(circuit_path))
        indices = gid2indices[gid]
        n_samples = len(indices)

        print(f"[{ci + 1}/{len(all_gids)}] gid={gid} circuit={circuit_name} "
              f"samples={n_samples} ... ", end="", flush=True)

        # --- encode graph ONCE, then free the graph from GPU ---
        g_circuit, _ = aag_to_dgl_graph(circuit_path)
        g_dev = g_circuit.to(device)
        h0 = g_dev.ndata["nf"].to(torch.float32)
        node_emb = model.gin(g_dev, h0)
        g_emb = model._graph_pool(g_dev, node_emb)  # (1, gin_hidden_dim)
        del g_dev, h0, node_emb, g_circuit
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- batch only sequences, reuse g_emb ---
        all_preds: List[torch.Tensor] = []
        all_trues: List[torch.Tensor] = []

        for batch_start in range(0, n_samples, args.batch_size):
            batch_idx = indices[batch_start : batch_start + args.batch_size]

            seqs: List[torch.Tensor] = []
            true_rows: List[List[float]] = []
            for idx in batch_idx:
                row = ds_eval.df.iloc[idx]
                seq_raw = load_seq(str(row[ds_eval.seq_col]))
                if not torch.is_tensor(seq_raw):
                    seq_raw = torch.tensor(seq_raw)
                seqs.append(seq_raw.to(torch.float32))
                true_rows.append([float(row[ds_eval.label2col[lb]]) for lb in task_names])

            seq_pad, seq_len = pad_sequences(seqs)
            seq_pad = seq_pad.to(device)
            seq_len = seq_len.to(device)

            out = _forward_with_cached_g_emb(model, g_emb, seq_pad, seq_len)
            pred = _extract_pred(out)
            if normalizer is not None:
                pred = normalizer.denormalize(pred)

            true = torch.tensor(true_rows, dtype=torch.float32)

            all_preds.append(pred.detach().cpu())
            all_trues.append(true)

        del g_emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pred_t = torch.cat(all_preds, dim=0)
        true_t = torch.cat(all_trues, dim=0)
        m = compute_metrics_original_space(pred_t, true_t, task_names)

        row = [gid, circuit_name, n_samples]
        msg_parts = []
        for i, lb in enumerate(task_names):
            r2 = float(m["r2_per_task"][i])
            mape = float(m["mape_per_task"][i]) * 100.0
            row.extend([r2, mape])
            msg_parts.append(f"{lb}: R2={r2:.6f}, MAPE={mape:.4f}%")
        print(" | ".join(msg_parts))
        rows.append(row)

    out_csv = args.output_csv
    headers = ["gid", "circuit", "n_samples"]
    for lb in task_names:
        headers.extend([f"{lb}_r2", f"{lb}_mape_percent"])

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"\n[OK] Saved per-circuit metrics to: {out_csv}")
    if len(ckpt_meta) > 0:
        print(f"[INFO] checkpoint meta keys: {sorted(list(ckpt_meta.keys()))}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference + per-circuit R2/MAPE report.")
    parser.add_argument("--ckpt-path", type=str, default="./ckpt/best_r2_by_circuit_output_large_area_delay.pt")
    parser.add_argument("--normalizer-path", type=str, default="")
    parser.add_argument("--csv-path", type=str, default = '/home/yfdai/asap/data/output_large.csv')
    parser.add_argument("--circuit-dir", type=str, default='/home/yfdai/asap/data/aag/')
    parser.add_argument("--seq-dir", type=str, default='/home/yfdai/asap/data/seq_large_new/')
    parser.add_argument("--tasks", type=str, default="area,delay")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--device", type=str)
    parser.add_argument("--strict", action="store_true", help="Use strict=True for load_state_dict.")
    parser.add_argument("--split", type=str, default="all", choices=["all", "by_circuit", "within_circuit"])
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", type=str, default="./output/large.csv")
    return parser


if __name__ == "__main__":
    run_inference(build_parser().parse_args())
