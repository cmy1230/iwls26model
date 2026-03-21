"""
在无需重新训练的前提下，加载 `finetune_lora.py` 产出的 LoRA 权重，对目标电路重新评估。

依赖目录中至少包含（由新版 finetune_lora 在训练结束时写出）：
  - lora_finetuned_model.pt   # state_dict + meta
  - lora_run_meta.json        # 与 .pt 内 meta 一致，便于查看路径

若你只有旧版输出（仅有 lora_finetune_metrics.json），需用新版 finetune_lora 重新跑一遍微调以生成上述文件。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from dataset_loader import collate_circuit_seq
from finetune_lora import (
    LABEL_NAMES,
    _metrics_to_jsonable,
    _try_load_g_emb_cache_file,
    _unwrap_base_dataset,
    forward_collect_predictions,
    get_circuit_dataset,
    precompute_gid_g_emb,
    setup_lora,
)
from label_normalizer import LabelNormalizer, compute_metrics_per_circuit
from model import TopCircuitSeqModel, TopCircuitSeqModelCfg


def _pearson_r_per_task(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-8) -> List[float]:
    """推理后在原始空间上对 pred vs true 逐任务计算 Pearson r（全样本 pooled）。"""
    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    if true.ndim == 3 and true.shape[-1] == 1:
        true = true.squeeze(-1)
    out: List[float] = []
    for t in range(pred.shape[1]):
        yt = true[:, t].float()
        pt = pred[:, t].float()
        yt_c = yt - yt.mean()
        pt_c = pt - pt.mean()
        denom = torch.sqrt((yt_c ** 2).sum() * (pt_c ** 2).sum()).clamp_min(eps)
        if float(denom.item()) < eps * 10:
            out.append(0.0)
        else:
            out.append(float((yt_c * pt_c).sum() / denom))
    return out


def _mask_top_fraction(
    y: torch.Tensor,
    frac: float,
    *,
    largest: bool,
) -> torch.Tensor:
    """真值维度上：前 frac 的样本（最大或最小 frac*n 个）。"""
    n = int(y.shape[0])
    k = max(1, int(math.ceil(n * frac)))
    _, idx = torch.sort(y, descending=largest)
    m = torch.zeros(n, dtype=torch.bool)
    m[idx[:k]] = True
    return m


def analyze_prune_largest_pred_area_vs_true_area_tier(
    pred: torch.Tensor,
    true: torch.Tensor,
    label_names: List[str],
    *,
    truth_area_largest: bool = False,
) -> Dict[str, Any]:
    """
    策略：剪掉「预测 area 最大」的样本（pred_area 降序取前 k 个）。

    统计误伤：真实 area **最小** 的 10% / 5% 档（默认）里，有多少比例会被这次剪枝误剪掉。
    误伤率 = |剪枝 ∩ 真实tier| / |真实tier|（百分数）。

    真实 tier 默认：真 area **最小** 的 ceil(n*10%)、ceil(n*5%) 个样本；可改 --truth_tier_area largest。
    """
    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    if true.ndim == 3 and true.shape[-1] == 1:
        true = true.squeeze(-1)

    name_to_i = {n: i for i, n in enumerate(label_names)}
    ia = name_to_i.get("area", 0)
    pa = pred[:, ia]
    ta = true[:, ia]
    n = int(pa.shape[0])

    tier10 = _mask_top_fraction(ta, 0.10, largest=truth_area_largest)
    tier5 = _mask_top_fraction(ta, 0.05, largest=truth_area_largest)
    _, idx_desc = torch.sort(pa, descending=True)

    scenarios: List[Dict[str, Any]] = []
    for prune_frac, desc_zh, key in (
        (0.50, "剪掉预测 area 最大的 50% 样本", "prune_largest_pred_area_50pct"),
        (0.75, "剪掉预测 area 最大的 75% 样本", "prune_largest_pred_area_75pct"),
        (0.80, "剪掉预测 area 最大的 80% 样本", "prune_largest_pred_area_80pct"),
    ):
        k = max(1, int(math.ceil(n * prune_frac)))
        prune = torch.zeros(n, dtype=torch.bool)
        prune[idx_desc[:k]] = True

        def hit_pct(tier_mask: torch.Tensor) -> Tuple[float, int, int]:
            nt = int(tier_mask.sum().item())
            if nt == 0:
                return 0.0, 0, 0
            h = int((prune & tier_mask).sum().item())
            return 100.0 * h / nt, h, nt

        p10, h10, n10 = hit_pct(tier10)
        p5, h5, n5 = hit_pct(tier5)
        scenarios.append(
            {
                "key": key,
                "description_zh": desc_zh,
                "prune_frac": prune_frac,
                "n_pruned": k,
                "n_total": n,
                "truth_area_tier": "largest" if truth_area_largest else "smallest",
                "pct_true_area_top10_mispruned": p10,
                "pct_true_area_top5_mispruned": p5,
                "counts": {
                    "true_top10_in_prune": h10,
                    "true_top10_n": n10,
                    "true_top5_in_prune": h5,
                    "true_top5_n": n5,
                },
            }
        )

    return {
        "prune_rule": "按预测 area 降序剪枝（剪掉预测最大的样本）",
        "truth_reference": "默认：真实 area 最小的 10% 与 5% 档（ceil(n*frac) 个）；可选最大档",
        "metric": "误伤率(%) = 该真实 tier 内被剪掉的样本数 / 该 tier 样本数",
        "scenarios": scenarios,
    }


def _print_prune_pred_area_analysis(a: Dict[str, Any]) -> None:
    print(
        "\n[prune_pred_area] 剪枝对象：预测 area **最大** 的若干样本；参照 tier：真实 area **最小** 的 10%/5% 档",
        flush=True,
    )
    t0 = a["scenarios"][0].get("truth_area_tier", "smallest")
    print(
        f"  真实 area 参照: 真值{'最大' if t0 == 'largest' else '最小'} 的 10% 与 5% 样本（--truth_tier_area largest|smallest）",
        flush=True,
    )
    lab10 = "真最小10%档" if t0 != "largest" else "真最大10%档"
    lab5 = "真最小5%档" if t0 != "largest" else "真最大5%档"
    for row in a["scenarios"]:
        p10 = row["pct_true_area_top10_mispruned"]
        p5 = row["pct_true_area_top5_mispruned"]
        c = row["counts"]
        print(
            f"  {row['description_zh']} (剪 n={row['n_pruned']}/{row['n_total']})",
            flush=True,
        )
        print(
            f"    误伤{lab10}: {p10:.2f}%  ({c['true_top10_in_prune']}/{c['true_top10_n']})"
            f"  |  误伤{lab5}: {p5:.2f}%  ({c['true_top5_in_prune']}/{c['true_top5_n']})",
            flush=True,
        )
    six: List[str] = []
    for row in a["scenarios"]:
        six.append(f"{row['pct_true_area_top10_mispruned']:.2f}%")
        six.append(f"{row['pct_true_area_top5_mispruned']:.2f}%")
    print(
        "  [汇总·6项误伤率] 剪50%最大预测→(最小10%档,最小5%档) | 剪75%→… | 剪80%→… : "
        + " | ".join(
            [
                f"50%:({six[0]},{six[1]})",
                f"75%:({six[2]},{six[3]})",
                f"80%:({six[4]},{six[5]})",
            ]
        ),
        flush=True,
    )


def _merge_inference_metrics(
    pred: torch.Tensor,
    true: torch.Tensor,
    gids: torch.Tensor,
) -> Dict[str, Any]:
    """一次前向后的指标：含 per-circuit 汇总，并强制写入推理阶段计算的 Pearson。"""
    m = compute_metrics_per_circuit(pred, true, gids, LABEL_NAMES)
    pr = _pearson_r_per_task(pred, true)
    m["pearson_per_task"] = pr
    m["pearson_mean"] = sum(pr) / max(1, len(pr))
    m["label_names"] = list(m.get("label_names") or LABEL_NAMES)
    return m


def _cfg_from_dict(d: Dict[str, Any]) -> TopCircuitSeqModelCfg:
    return TopCircuitSeqModelCfg(
        num_tasks=d["num_tasks"],
        gin_in_dim=d["gin_in_dim"],
        gin_hidden_dim=d["gin_hidden_dim"],
        gin_layers=d["gin_layers"],
        seq_in_dim=d["seq_in_dim"],
        seq_hidden_dim=d["seq_hidden_dim"],
        seq_layers=d["seq_layers"],
        fe_num_heads=d["fe_num_heads"],
        fe_num_layers=d["fe_num_layers"],
        fs_num_heads=d["fs_num_heads"],
        fs_num_layers=d["fs_num_layers"],
        ens_num_classes=d["ens_num_classes"],
        ens_num_layers=d["ens_num_layers"],
        ens_hidden_dim=d["ens_hidden_dim"],
        fs_dropout=d.get("fs_dropout", 0.0),
        fs_attn_dropout=d.get("fs_attn_dropout", 0.0),
        ens_dropout=d.get("ens_dropout", 0.0),
        ens_attn_dropout=d.get("ens_attn_dropout", 0.0),
        graph_pool=d.get("graph_pool", "mean"),
    )


def _merge_path(meta_val: str, override: Optional[str]) -> str:
    if override and str(override).strip():
        return os.path.abspath(str(override).strip())
    return (meta_val or "").strip()


def _build_loader(
    circuit_ds: Any,
    split: str,
    train_indices: List[int],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    n = len(circuit_ds)
    tr_set: Set[int] = set(int(i) for i in train_indices)
    if split == "full":
        ds = circuit_ds
    elif split == "train":
        ds = Subset(circuit_ds, sorted(tr_set))
    elif split == "test":
        te = sorted(set(range(n)) - tr_set)
        ds = Subset(circuit_ds, te)
    else:
        raise ValueError(f"Unknown split={split!r}")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_circuit_seq,
    )


def _print_metrics(m: Dict[str, Any]) -> None:
    r2 = m.get("r2_avg_over_circuits", m.get("r2_mean", 0.0))
    mape = m.get("mape_avg_over_circuits", m.get("mape_mean", 0.0))
    pr_mean = m.get("pearson_mean", 0.0)
    print(f"  R2_avg_over_circuits={r2:.4f} MAPE_avg_over_circuits={mape*100:.2f}% mse={m.get('mse', 0):.6f}")
    print(f"  Pearson_r (pooled, 推理计算): mean={pr_mean:.4f}", flush=True)
    r2_pt = m.get("r2_per_task") or []
    mape_pt = m.get("mape_per_task") or []
    pr_pt = m.get("pearson_per_task") or []
    names = m.get("label_names") or LABEL_NAMES
    for i, name in enumerate(names):
        if i < len(r2_pt) and i < len(mape_pt):
            pr_s = f" Pearson_r={pr_pt[i]:.4f}" if i < len(pr_pt) else ""
            print(f"    {name}: R2={r2_pt[i]:.4f} MAPE={mape_pt[i]*100:.2f}% {pr_s}".rstrip())


def main() -> None:
    p = argparse.ArgumentParser(
        description="加载 LoRA 微调产物并评估（area/delay）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
必填：
  --lora_run_dir  内含 lora_finetuned_model.pt（由 finetune_lora 生成）

可选覆盖（数据换机器或路径变了时）：--csv --circuit_dir --seq_dir --circuit_name
  --normalizer_path --graph_emb_cache_path

还需要什么（已写入 lora_run_meta / .pt 内 meta，一般不必重复传）：
  - 与微调时相同的 CSV、AAG 目录、序列目录、电路名子串
  - label 归一化 JSON（normalizer）
  - 大图建议提供 graph_emb_cache（与微调时同路径或重新预计算）

剪枝分析（默认开启）：
  剪掉「预测 area 最大」的 50%%/75%%/80%% 样本；误伤参照「真实 area 最小」的 10%% 与 5%% 档（可调 largest）。
""",
    )
    p.add_argument(
        "--lora_run_dir",
        type=str,
        default="",
        help="finetune_lora 的 output_dir，需含 lora_finetuned_model.pt",
    )
    p.add_argument(
        "--lora_weights",
        type=str,
        default="",
        help="可选，直接指定 .pt 路径（默认 <lora_run_dir>/lora_finetuned_model.pt）",
    )
    p.add_argument("--csv", type=str, default="", help="覆盖 meta 中的 csv")
    p.add_argument("--circuit_dir", type=str, default="", help="覆盖 meta 中的 circuit_dir")
    p.add_argument("--seq_dir", type=str, default="", help="覆盖 meta 中的 seq_dir")
    p.add_argument("--circuit_name", type=str, default="", help="覆盖 meta 中的 circuit_name")
    p.add_argument("--normalizer_path", type=str, default="", help="覆盖 meta 中的 normalizer_path")
    p.add_argument("--graph_emb_cache_path", type=str, default="", help="覆盖 meta 中的图向量缓存")
    p.add_argument(
        "--no_graph_emb_cache",
        action="store_true",
        help="禁用缓存，每 batch 重算 GIN（慢，易 OOM）",
    )
    p.add_argument(
        "--split",
        type=str,
        choices=["full", "test", "train"],
        default="test",
        help="评估子集：与微调时相同的 train 划分下的 test / train，或 full（含训练行）",
    )
    p.add_argument("--batch_size", type=int, default=0, help="0 表示用 meta 中的 batch_size")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="")
    p.add_argument(
        "--output_json",
        type=str,
        default="",
        help="将指标写入该路径（JSON）",
    )
    p.add_argument(
        "--truth_tier_area",
        type=str,
        choices=["largest", "smallest"],
        default="smallest",
        help="「真实 area 10%%/5%% 档」按真值最大或最小取（默认 smallest=真 area 最小的 10%%/5%%）",
    )
    p.add_argument(
        "--skip_pred_area_prune_analysis",
        action="store_true",
        help="不计算「按预测 area 剪枝」对真实 area tier 的误伤率",
    )
    args = p.parse_args()

    lora_pt = args.lora_weights.strip()
    if not lora_pt:
        if not args.lora_run_dir.strip():
            p.error("请提供 --lora_run_dir 或 --lora_weights")
        lora_pt = os.path.join(args.lora_run_dir.strip(), "lora_finetuned_model.pt")
    if not os.path.isfile(lora_pt):
        raise FileNotFoundError(
            f"未找到 LoRA 权重: {lora_pt}\n"
            "请使用新版 finetune_lora 重新微调以生成 lora_finetuned_model.pt。"
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(lora_pt, map_location="cpu")
    if not isinstance(blob, dict) or "state_dict" not in blob or "meta" not in blob:
        raise ValueError(f"无效的 LoRA 文件格式: {lora_pt}")
    meta: Dict[str, Any] = blob["meta"]

    csv_path = _merge_path(meta.get("csv", ""), args.csv or None)
    circuit_dir = _merge_path(meta.get("circuit_dir", ""), args.circuit_dir or None)
    seq_dir = _merge_path(meta.get("seq_dir", ""), args.seq_dir or None)
    circuit_name = (args.circuit_name.strip() if args.circuit_name.strip() else None) or meta.get(
        "circuit_name", ""
    )
    norm_path = _merge_path(meta.get("normalizer_path", ""), args.normalizer_path or None)
    gcache_arg = args.graph_emb_cache_path.strip()
    gcache_meta = (meta.get("graph_emb_cache_path") or "").strip()
    cache_path = gcache_arg if gcache_arg else gcache_meta
    no_cache = args.no_graph_emb_cache or bool(meta.get("no_graph_emb_cache"))

    for label, val in [
        ("csv", csv_path),
        ("circuit_dir", circuit_dir),
        ("seq_dir", seq_dir),
        ("circuit_name", circuit_name),
        ("normalizer_path", norm_path),
    ]:
        if not val:
            raise ValueError(f"缺少 {label}（meta 中无有效值且未通过命令行覆盖）")

    batch_size = args.batch_size if args.batch_size > 0 else int(meta.get("batch_size", 4))

    circuit_ds = get_circuit_dataset(
        csv_path=csv_path,
        circuit_name=str(circuit_name),
        circuit_dir=circuit_dir,
        seq_dir=seq_dir,
    )
    train_indices: List[int] = list(meta.get("train_indices") or [])
    if not train_indices:
        raise ValueError("meta 中缺少 train_indices，无法用新版脚本复现 train/test 划分。")

    normalizer: Optional[LabelNormalizer] = None
    if norm_path:
        normalizer = LabelNormalizer.load(norm_path)

    model_cfg = _cfg_from_dict(meta["model_cfg"])
    model = TopCircuitSeqModel(model_cfg).to(device)
    setup_lora(model, int(meta["lora_r"]), float(meta["lora_alpha"]))
    model.load_state_dict(blob["state_dict"], strict=True)
    model.eval()

    gin_h = int(meta.get("gin_hidden_dim", model_cfg.gin_hidden_dim))
    g_emb_cache = None
    if not no_cache:
        base_ds = _unwrap_base_dataset(circuit_ds)
        need_gids: Set[int] = {int(x) for x in circuit_ds.df["gid"].tolist()}
        gids_sorted = sorted(need_gids)
        loaded = _try_load_g_emb_cache_file(cache_path, gin_h, need_gids) if cache_path else None
        if loaded is not None:
            g_emb_cache = loaded
            print(f"[g_emb] loaded from {cache_path}", flush=True)
        else:
            print(f"[g_emb] precomputing GIN+pool for gids={gids_sorted} ...", flush=True)
            g_emb_cache = precompute_gid_g_emb(model, base_ds, gids_sorted, device)
            print("[g_emb] precompute done", flush=True)
    else:
        print("[g_emb] cache disabled (--no_graph_emb_cache)", flush=True)

    loader = _build_loader(
        circuit_ds,
        args.split,
        train_indices,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )
    print(
        f"[eval] split={args.split} n={len(loader.dataset)} circuit={circuit_name!r}",
        flush=True,
    )
    pred, true, gids = forward_collect_predictions(
        model, loader, device, normalizer, g_emb_cache=g_emb_cache
    )
    metrics = _merge_inference_metrics(pred, true, gids)
    print("[metrics]")
    _print_metrics(metrics)

    prune_analysis: Optional[Dict[str, Any]] = None
    if not args.skip_pred_area_prune_analysis:
        prune_analysis = analyze_prune_largest_pred_area_vs_true_area_tier(
            pred,
            true,
            list(metrics.get("label_names") or LABEL_NAMES),
            truth_area_largest=(args.truth_tier_area == "largest"),
        )
        _print_prune_pred_area_analysis(prune_analysis)

    if args.output_json.strip():
        out_path = os.path.abspath(args.output_json.strip())
        payload: Dict[str, Any] = {
            "lora_weights": os.path.abspath(lora_pt),
            "split": args.split,
            "circuit_name": circuit_name,
            "metrics": _metrics_to_jsonable(metrics),
        }
        if prune_analysis is not None:
            payload["pred_area_prune_vs_true_area_tier"] = _metrics_to_jsonable(prune_analysis)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
