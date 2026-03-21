import os
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterator, Literal

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import dgl

import time, traceback, os


from aig_preprocessing import load_aig_as_dgl
from aig_preprocess_seq import aag_to_dgl_graph
from seq_preprocessing import load_seq


# -----------------------------
# Helpers
# -----------------------------
def _resolve_path(root_dir: Optional[str], p: str) -> str:
    """
    If p is absolute: return as-is.
    Else if root_dir is provided: join(root_dir, p).
    Else: return p.
    """
    p = str(p)
    if os.path.isabs(p):
        return p
    if root_dir is None or str(root_dir).strip() == "":
        return p
    return os.path.join(root_dir, p)


def _resolve_circuit_path(circuit_dir: Optional[str], p: str) -> str:
    p = str(p)

    # 如果没有 .aig 后缀，就补上
    if not p.endswith(".aag"):
        p = p + ".aag"

    # 再处理路径
    if os.path.isabs(p):
        return p
    if circuit_dir is None or circuit_dir.strip() == "":
        return p
    return os.path.join(circuit_dir, p)


def pad_sequences(seqs: List[torch.Tensor], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    seqs: list of (L_i, D)
    returns:
      padded: (B, Lmax, D)
      lengths: (B,)
    """
    if len(seqs) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long)

    lengths = torch.tensor([int(s.shape[0]) for s in seqs], dtype=torch.long)
    max_len = int(lengths.max().item())
    feat_dim = int(seqs[0].shape[1])

    padded = seqs[0].new_full((len(seqs), max_len, feat_dim), fill_value=pad_value)
    for i, s in enumerate(seqs):
        L = int(s.shape[0])
        padded[i, :L, :] = s
    return padded, lengths


# -----------------------------
# Dataset
# -----------------------------
SupportedLabel = Literal["nd", "area", "delay", "lev", "runtime"]


@dataclass
class Sample:
    g: dgl.DGLGraph
    seq: torch.Tensor                      # (L, Dseq)
    y: Dict[str, torch.Tensor]             # {"delay": scalar, "nd": scalar, ...}
    gid: int = 0                           # circuit id for per-circuit metrics
    y_cls: Optional[torch.Tensor] = None   # optional


class CircuitSeqDataset(Dataset):
    """
    CSV columns (by default):
      0: seq_path
      1: circuit_path
      2: status       ("Failed" will be filtered out)
      3: nd
      4: area
      5: delay
      6: lev
      7: runtime

    Important:
      - Supports circuit_dir / seq_dir to resolve relative paths.
      - Optimization 1: preload graphs, build gid column, graph_pool shared across subsets.
      - NEW: configurable regression labels via labels=[...]
      - Optional ``df``: 若传入，则不再读 CSV；df 须已等价于内部「去 Failed + resolve 路径」后的结果；
        用于只 preload 子集涉及电路的 AAG（如单电路微调）。
    """

    def __init__(
        self,
        csv_path: str,
        *,
        circuit_dir: Optional[str] = None,
        seq_dir: Optional[str] = None,
        use_header: bool = True,
        check_paths: bool = False,
        preload_graphs: bool = True,
        # 若提供，则不再 read_csv；df 须已通过「去 Failed + 路径 resolve」处理（与 self.df 一致）
        df: Optional["pd.DataFrame"] = None,

        # if CSV has header and you want explicit col names, you can set these:
        circuit_col: Optional[str] = None,
        seq_col: Optional[str] = None,
        status_col: Optional[str] = None,
        nd_col: Optional[str] = None,
        area_col: Optional[str] = None,
        delay_col: Optional[str] = None,
        lev_col: Optional[str] = None,
        runtime_col: Optional[str] = None,

        # NEW: choose which regression labels to output
        labels: Optional[List[SupportedLabel]] = None,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.circuit_dir = circuit_dir
        self.seq_dir = seq_dir

        input_df_provided = df is not None
        if df is not None:
            df = df.copy().reset_index(drop=True)
            if "gid" in df.columns:
                df = df.drop(columns=["gid"])
        else:
            df = pd.read_csv(csv_path, header=0 if use_header else None)

        # decide columns
        if use_header:
            if seq_col is None:     seq_col = df.columns[0]
            if circuit_col is None: circuit_col = df.columns[1]
            if status_col is None:  status_col = df.columns[2]
            if nd_col is None:      nd_col = df.columns[3]
            if area_col is None:    area_col = df.columns[4]
            if delay_col is None:   delay_col = df.columns[5]
            if lev_col is None:     lev_col = df.columns[6]
            if runtime_col is None: runtime_col = df.columns[7]
        else:
            seq_col, circuit_col, status_col = 0, 1, 2
            nd_col, area_col, delay_col, lev_col, runtime_col = 3, 4, 5, 6, 7

        self.circuit_col = circuit_col
        self.seq_col = seq_col
        self.status_col = status_col
        self.nd_col = nd_col
        self.area_col = area_col
        self.delay_col = delay_col
        self.lev_col = lev_col
        self.runtime_col = runtime_col

        # -----------------------------
        # which labels?
        # -----------------------------
        supported: List[str] = ["nd", "area", "delay", "lev", "runtime"]
        if labels is None:
            labels = ["nd", "area", "delay", "lev", "runtime"]
        for lb in labels:
            if lb not in supported:
                raise ValueError(f"Unknown label '{lb}'. Supported labels: {supported}")
        self.labels: List[str] = list(labels)

        # map label name -> df column
        self.label2col: Dict[str, Any] = {
            "nd": self.nd_col,
            "area": self.area_col,
            "delay": self.delay_col,
            "lev": self.lev_col,
            "runtime": self.runtime_col,
        }

        # -----------------------------
        # filter Failed + resolve paths（若 df 由外部传入，则视为已处理好，跳过）
        # -----------------------------
        if input_df_provided:
            self.df = df
        else:
            status_series = df[self.status_col].astype(str).str.strip().str.lower()
            df_ok = df[status_series != "failed"].reset_index(drop=True)

            df_ok = df_ok.copy()
            df_ok[self.circuit_col] = df_ok[self.circuit_col].astype(str).map(
                lambda p: _resolve_circuit_path(self.circuit_dir, p)
            )
            df_ok[self.seq_col] = df_ok[self.seq_col].astype(str).map(
                lambda p: _resolve_path(self.seq_dir, p)
            )
            self.df = df_ok

        # optional path sanity check (sample a few rows)
        if check_paths and len(self.df) > 0:
            for idx in [0, len(self.df) // 2, len(self.df) - 1]:
                cpath = str(self.df.iloc[idx][self.circuit_col])
                spath = str(self.df.iloc[idx][self.seq_col])
                if not os.path.exists(cpath):
                    raise FileNotFoundError(f"circuit_path not found: {cpath}")
                if not os.path.exists(spath):
                    raise FileNotFoundError(f"seq_path not found: {spath}")

        # -----------------------------
        # Optimization 1: graph_pool + gid
        # -----------------------------
        self.circuits: List[str] = self.df[self.circuit_col].astype(str).unique().tolist()
        self.circuit2gid: Dict[str, int] = {p: i for i, p in enumerate(self.circuits)}
        self.df["gid"] = self.df[self.circuit_col].astype(str).map(self.circuit2gid).astype(int)

        self.graph_pool: Optional[List[dgl.DGLGraph]] = None
        if preload_graphs:
            print(f"[preload] loading {len(self.circuits)} graphs...", flush=True)
            self.graph_pool = [None] * len(self.circuits)
            # for p, gid in self.circuit2gid.items():
            #     print(f"[preload] gid={gid} path={p}", flush=True)
            #     # self.graph_pool[gid] = load_aig_as_dgl(p)
            #     self.graph_pool[gid], _ = aag_to_dgl_graph(p)


            for p, gid in self.circuit2gid.items():
                t0 = time.time()
                try:
                    size_mb = os.path.getsize(p) / (1024 * 1024)
                    print(f"[preload] gid={gid} size={size_mb:.1f}MB path={p}", flush=True)

                    g, extra = aag_to_dgl_graph(p)

                    dt = time.time() - t0
                    print(f"[preload] gid={gid} done in {dt:.2f}s | nodes={g.num_nodes()} edges={g.num_edges()}", flush=True)
                    self.graph_pool[gid] = g

                except Exception as e:
                    print(f"[preload] gid={gid} FAILED: {e}", flush=True)
                    traceback.print_exc()
                    # 你可以选择 continue，或者 raise 让它直接报错停下
                    raise


    def make_subset(self, df_subset: pd.DataFrame) -> "CircuitSeqDataset":
        """
        Create a subset dataset sharing graph_pool/circuit mapping.
        This is used by your separate "split strategy" file.
        """
        ds = object.__new__(CircuitSeqDataset)

        ds.csv_path = self.csv_path
        ds.circuit_dir = self.circuit_dir
        ds.seq_dir = self.seq_dir

        ds.circuit_col = self.circuit_col
        ds.seq_col = self.seq_col
        ds.status_col = self.status_col
        ds.nd_col = self.nd_col
        ds.area_col = self.area_col
        ds.delay_col = self.delay_col
        ds.lev_col = self.lev_col
        ds.runtime_col = self.runtime_col

        ds.labels = self.labels
        ds.label2col = self.label2col

        ds.df = df_subset.reset_index(drop=True)

        ds.circuits = self.circuits
        ds.circuit2gid = self.circuit2gid
        ds.graph_pool = self.graph_pool

        return ds

    def __len__(self) -> int:
        return len(self.df)

    def _get_graph(self, gid: int) -> dgl.DGLGraph:
        if self.graph_pool is None:
            g, _ = aag_to_dgl_graph(self.circuits[gid])
            return g
        return self.graph_pool[gid]

    def _load_seq(self, seq_path: str) -> torch.Tensor:
        seq = load_seq(seq_path)  # (L, Dseq)
        if not torch.is_tensor(seq):
            seq = torch.tensor(seq)
        return seq.to(torch.float32)

    def __getitem__(self, idx: int) -> Sample:
        row = self.df.iloc[idx]

        gid = int(row["gid"])
        seq_path = str(row[self.seq_col])

        g = self._get_graph(gid)
        seq = self._load_seq(seq_path)

        y: Dict[str, torch.Tensor] = {}
        for lb in self.labels:
            col = self.label2col[lb]
            y[lb] = torch.tensor(float(row[col]), dtype=torch.float32)  # scalar

        return Sample(g=g, seq=seq, y=y, gid=gid, y_cls=None)


# -----------------------------
# Collate function
# -----------------------------
def collate_circuit_seq(samples: List[Sample]) -> Dict[str, Any]:
    """
    Returns a batch dict:
      - g: batched DGLGraph
      - seq: (B, Lmax, Dseq) padded
      - seq_len: (B,)
      - each selected label: (B,)  e.g., batch["delay"], batch["nd"], ...
    """
    if len(samples) == 0:
        raise ValueError("Empty batch: samples is empty.")

    graphs = [s.g for s in samples]
    seqs = [s.seq for s in samples]

    bg = dgl.batch(graphs)
    seq_pad, seq_len = pad_sequences(seqs, pad_value=0.0)

    batch: Dict[str, Any] = {
        "g": bg,
        "seq": seq_pad,
        "seq_len": seq_len,
    }

    # labels: each becomes its own key
    label_names = list(samples[0].y.keys())
    for lb in label_names:
        batch[lb] = torch.stack([s.y[lb] for s in samples], dim=0)  # (B,)

    # gid for per-circuit metrics
    batch["gid"] = torch.tensor([s.gid for s in samples], dtype=torch.long)  # (B,)

    # optional y_cls
    if samples[0].y_cls is not None:
        y0 = samples[0].y_cls
        if torch.is_tensor(y0) and y0.ndim == 0:
            batch["y_cls"] = torch.stack([s.y_cls for s in samples], dim=0)  # (B,)
        else:
            batch["y_cls"] = torch.stack([s.y_cls for s in samples], dim=0)  # (B,C)

    return batch


# -----------------------------
# Optimization 2: Group-by-gid BatchSampler
# -----------------------------
class GroupByGIDBatchSampler(Sampler[List[int]]):
    """
    Make each batch (as much as possible) come from the same gid (same circuit).
    dataset.df must have a 'gid' column.
    """

    def __init__(self, dataset: CircuitSeqDataset, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

        gids = dataset.df["gid"].tolist()
        self.gid2indices: Dict[int, List[int]] = {}
        for i, gid in enumerate(gids):
            self.gid2indices.setdefault(int(gid), []).append(i)
        self.all_gids = sorted(self.gid2indices.keys())

    def __iter__(self) -> Iterator[List[int]]:
        gids = self.all_gids[:]
        if self.shuffle:
            random.shuffle(gids)

        for gid in gids:
            idxs = self.gid2indices[gid][:]
            if self.shuffle:
                random.shuffle(idxs)

            for start in range(0, len(idxs), self.batch_size):
                batch = idxs[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        total = 0
        for gid in self.all_gids:
            n = len(self.gid2indices[gid])
            total += (n // self.batch_size) if self.drop_last else math.ceil(n / self.batch_size)
        return total


# -----------------------------
# DataLoader builder
# -----------------------------
def build_dataloader(
    csv_path: str,
    batch_size: int,
    *,
    circuit_dir: Optional[str] = None,
    seq_dir: Optional[str] = None,
    labels: Optional[List[SupportedLabel]] = None,  # NEW
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    use_header: bool = True,
    group_by_gid: bool = True,
    drop_last: bool = False,
    preload_graphs: bool = True,
) -> DataLoader:
    ds = CircuitSeqDataset(
        csv_path,
        circuit_dir=circuit_dir,
        seq_dir=seq_dir,
        use_header=use_header,
        check_paths=False,
        preload_graphs=preload_graphs,
        labels=labels,
    )

    if group_by_gid:
        batch_sampler = GroupByGIDBatchSampler(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        dl = DataLoader(
            ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_circuit_seq,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
    else:
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_circuit_seq,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            drop_last=drop_last,
        )

    return dl


# -----------------------------
# Quick sanity check
# -----------------------------
if __name__ == "__main__":
    dl = build_dataloader(
        csv_path="/home/yfdai/asap/data/test.csv",
        circuit_dir="/home/yfdai/asap/data/aag",
        seq_dir="/home/yfdai/asap/data/seq/",
        batch_size=8,
        shuffle=True,
        num_workers=1,
        group_by_gid=True,
        preload_graphs=True,
        labels=["delay", "area"],  # choose any subset
    )

    batch = next(iter(dl))
    print("batched graph:", batch["g"])
    print("seq shape:", batch["seq"].shape)
    print("seq_len shape:", batch["seq_len"].shape)
    for k in ["delay", "area"]:
        print(f"{k} shape:", batch[k].shape)  # (B,)
