"""代理模型辅助：将 bayesian_search.ACTIONS 映射为模型输入 (L, D)。"""

import csv
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from bayesian_search import NOP
from infer_per_circuit import _build_model, _extract_pred, _forward_with_cached_g_emb
from label_normalizer import LabelNormalizer
from model import load_pt_checkpoint
from seq_preprocessing import DEFAULT_COMMANDS


def _norm_header_key(k: Any) -> str:
    """表头或行键：去 BOM、首尾空白，便于与 'circuit' 等列名匹配。"""
    if k is None:
        return ""
    return str(k).strip().lstrip("\ufeff").lower()


def _row_get_ci(row: Dict[str, Any], key: str) -> Any:
    """大小写不敏感、容忍表头 BOM（\\ufeffcircuit）的列取值。"""
    want = _norm_header_key(key)
    for rk, rv in row.items():
        if _norm_header_key(rk) == want:
            return rv
    return None


class ActionEncoder:
    """ACTIONS 索引序列 → 子命令 one-hot 序列 (L, len(DEFAULT_COMMANDS))。"""

    def __init__(self, actions: List[str]):
        self._cmd2idx = {c: i for i, c in enumerate(DEFAULT_COMMANDS)}
        self._D = len(DEFAULT_COMMANDS)
        self._action_rows: List[List[int]] = []
        for a in actions:
            self._action_rows.append(self._parse_action(a))

    def _parse_action(self, action_str: str) -> List[int]:
        if action_str == NOP:
            return []
        s = action_str.strip()
        full_key = s if s.endswith(";") else s + ";"
        if full_key in self._cmd2idx:
            return [self._cmd2idx[full_key]]
        parts = [p.strip() for p in action_str.split(";") if p.strip()]
        rows: List[int] = []
        for p in parts:
            key = p + ";"
            if key in self._cmd2idx:
                rows.append(self._cmd2idx[key])
        return rows

    def encode(self, action_indices: List[int]) -> torch.Tensor:
        rows: List[torch.Tensor] = []
        for idx in action_indices:
            for cmd_idx in self._action_rows[idx]:
                v = torch.zeros(self._D, dtype=torch.float32)
                v[cmd_idx] = 1.0
                rows.append(v)
        if not rows:
            return torch.zeros(1, self._D, dtype=torch.float32)
        return torch.stack(rows, dim=0)


class ReliabilityChecker:
    """按 CSV 中误伤指标选取不超过阈值的最大安全剪枝比例。"""

    CUT_LEVELS = [
        ("hit_pct_of_truemin10_subset_predmax50", 0.50),
        ("hit_pct_of_truemin10_subset_predmax75", 0.75),
        ("hit_pct_of_truemin10_subset_predmax80", 0.80),
    ]
    MIS_HIT_THRESHOLD = 20.0  # 百分比，不超过 20% 误伤

    @staticmethod
    def check(circuit_name: str, csv_path: str, ckpt_dir: str) -> dict:
        result = {"enabled": False, "safe_cut_ratio": 0.0}

        ckpt_path = os.path.join(ckpt_dir, f"iwls26_{circuit_name}.pt")
        if not os.path.isfile(ckpt_path):
            print(f"[Surrogate] ckpt不存在: {ckpt_path}，禁用模型加速")
            return result

        if not csv_path or not os.path.isfile(csv_path):
            result["enabled"] = True
            result["safe_cut_ratio"] = 0.50
            return result

        row = None
        # utf-8-sig：去掉文件头 BOM，避免首列表名变成 "\\ufeffcircuit" 导致 KeyError
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if not any(_norm_header_key(n) == "circuit" for n in fieldnames):
                print(
                    "[Surrogate] CSV 无 circuit 列（检查表头拼写、或是否用 utf-8-sig/BOM），禁用模型加速"
                )
                return result
            for r in reader:
                c = _row_get_ci(r, "circuit")
                if c is not None and str(c).strip() == circuit_name:
                    row = r
                    break

        st = _row_get_ci(row, "status") if row is not None else None
        if row is None or str(st or "").strip() != "ok":
            print(f"[Surrogate] CSV中未找到{circuit_name}或状态非ok，禁用")
            return result

        safe_cut_ratio = 0.0
        for col, ratio in ReliabilityChecker.CUT_LEVELS:
            v = _row_get_ci(row, col)
            val = float(v) if v is not None and str(v).strip() != "" else 100.0
            if val < ReliabilityChecker.MIS_HIT_THRESHOLD:
                safe_cut_ratio = ratio

        if safe_cut_ratio == 0.0:
            print(f"[Surrogate] {circuit_name} 各剪枝比例均超过20%误伤阈值，禁用")
            return result

        result["enabled"] = True
        result["safe_cut_ratio"] = safe_cut_ratio
        print(f"[Surrogate] {circuit_name} 安全剪枝比={safe_cut_ratio*100:.0f}%")
        return result


class ModelSurrogate:
    def __init__(
        self,
        circuit_name: str,
        aag_path: str,
        ckpt_dir: str,
        actions: List[str],
        csv_path: str = "",
        device: str = "cpu",
    ):
        self.enabled = False
        self.safe_cut_ratio = 0.0
        self.device = torch.device(device)

        info = ReliabilityChecker.check(circuit_name, csv_path, ckpt_dir)
        if not info["enabled"]:
            return
        self.safe_cut_ratio = info["safe_cut_ratio"]

        ckpt_path = os.path.join(ckpt_dir, f"iwls26_{circuit_name}.pt")
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=self.device)

        num_bins = int(ckpt.get("num_bins", 8))
        self.model = _build_model(["area", "delay"], num_bins).to(self.device)
        load_pt_checkpoint(
            self.model, ckpt_path, map_location=self.device, strict=False, ckpt=ckpt
        )
        self.model.eval()

        norm_path = ckpt.get("normalizer_path", "")
        if isinstance(norm_path, str) and os.path.isfile(norm_path):
            self.normalizer = LabelNormalizer.load(norm_path)
        else:
            cand = os.path.join(ckpt_dir, f"iwls26_{circuit_name}_normalizer.json")
            self.normalizer = LabelNormalizer.load(cand) if os.path.isfile(cand) else None

        print(f"[Surrogate] 编码电路图: {aag_path} ...")
        from aig_preprocess_seq import aag_to_dgl_graph

        g, _ = aag_to_dgl_graph(aag_path)
        g = g.to(self.device)
        h0 = g.ndata["nf"].to(torch.float32)
        with torch.no_grad():
            node_emb = self.model.gin(g, h0)
            self.g_emb = self.model._graph_pool(g, node_emb)
        del g, h0, node_emb
        print(f"[Surrogate] 图编码完成，g_emb shape={tuple(self.g_emb.shape)}")

        self.encoder = ActionEncoder(actions)

        self.enabled = True
        print(f"[Surrogate] 启用成功，安全剪枝比={self.safe_cut_ratio*100:.0f}%")

    @torch.no_grad()
    def predict_batch(self, seqs: List[List[int]]) -> np.ndarray:
        """批量预测，返回 (N, 2) [area, delay] 原始空间"""
        from dataset_loader import pad_sequences

        tensors = [self.encoder.encode(s) for s in seqs]
        seq_pad, seq_len = pad_sequences(tensors)
        seq_pad = seq_pad.to(self.device)
        seq_len = seq_len.to(self.device)

        out = _forward_with_cached_g_emb(self.model, self.g_emb, seq_pad, seq_len)
        pred = _extract_pred(out)
        if self.normalizer is not None:
            pred = self.normalizer.denormalize(pred)
        return pred.cpu().numpy()

    def filter_candidates(
        self,
        candidates: List[List[int]],
        return_orig_indices: bool = False,
    ) -> Union[List[List[int]], Tuple[List[List[int]], np.ndarray]]:
        """按 safe_cut_ratio 剪掉预测最差的序列，返回保留的子集。不启用时原样返回。

        return_orig_indices=True 时额外返回 keep 对应的原列表下标（过滤后第 j 项来自
        candidates[orig_indices[j]]），供调用方在重排后仍识别某区段（如 rand 段）。
        """
        if not self.enabled or len(candidates) == 0:
            if return_orig_indices:
                return candidates, np.arange(len(candidates), dtype=np.int64)
            return candidates

        preds = self.predict_batch(candidates)
        products = preds[:, 0] * preds[:, 1]

        n_keep = max(1, int(len(candidates) * (1.0 - self.safe_cut_ratio)))
        keep_idx = np.argsort(products)[:n_keep]
        out = [candidates[i] for i in keep_idx]
        if return_orig_indices:
            return out, keep_idx.astype(np.int64, copy=False)
        return out
