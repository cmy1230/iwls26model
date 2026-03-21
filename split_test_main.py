# split_test_main.py
import os
from collections import Counter
from typing import List, Optional

import torch

from dataset_loader import CircuitSeqDataset, collate_circuit_seq, SupportedLabel
from split_dataset import split_dataset


def _gid_to_name(ds: CircuitSeqDataset, gid: int) -> str:
    """
    将 gid 映射成更易读的 circuit 名称：
    - 优先取 ds.circuits[gid] 的 basename（如 s1238.v.aig）
    """
    try:
        p = ds.circuits[int(gid)]
    except Exception:
        return f"gid={gid}"
    base = os.path.basename(str(p))
    return base


def _print_split_circuit_stats(tag: str, ds: CircuitSeqDataset) -> None:
    """
    打印该 split 中每个 circuit 的样本数。
    """
    if len(ds) == 0:
        print(f"\n[{tag}] EMPTY split.")
        return

    gids = ds.df["gid"].astype(int).tolist()
    cnt = Counter(gids)

    # 排序：按样本数降序
    items = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))

    print(f"\n[{tag}] samples={len(ds)}  unique_gids={len(items)}  labels={ds.labels}")
    for gid, n in items:
        name = _gid_to_name(ds, gid)
        print(f"  gid={gid:>3d}  n={n:>6d}  circuit={name}")


def _print_overlap_check(train_ds: CircuitSeqDataset, val_ds: CircuitSeqDataset, test_ds: CircuitSeqDataset) -> None:
    """
    within_circuit 模式下，理论上每个 gid 都应该在三份里都出现（除非该 gid 样本太少）。
    """
    train_g = set(train_ds.df["gid"].astype(int).unique())
    val_g = set(val_ds.df["gid"].astype(int).unique())
    test_g = set(test_ds.df["gid"].astype(int).unique())

    all_g = sorted(set.union(train_g, val_g, test_g))
    only_train = sorted(train_g - (val_g | test_g))
    only_val = sorted(val_g - (train_g | test_g))
    only_test = sorted(test_g - (train_g | val_g))

    print("\n[CHECK] gid overlap summary")
    print(f"  total unique gids: {len(all_g)}")
    print(f"  gids only in train: {len(only_train)} -> {only_train[:20]}{' ...' if len(only_train) > 20 else ''}")
    print(f"  gids only in val  : {len(only_val)} -> {only_val[:20]}{' ...' if len(only_val) > 20 else ''}")
    print(f"  gids only in test : {len(only_test)} -> {only_test[:20]}{' ...' if len(only_test) > 20 else ''}")

    inter = train_g & val_g & test_g
    print(f"  gids in all three : {len(inter)}")


def _test_labels_in_sample(ds: CircuitSeqDataset, tag: str) -> None:
    """
    测试 labels 是否正确传递到 subset 并能正常获取样本。
    注意：这里打印的是“单个样本”的 label 值，不是均值。
    """
    if len(ds) == 0:
        print(f"\n[{tag}] EMPTY - skip label test.")
        return

    print(f"\n[{tag}] Testing labels: {ds.labels}")
    print("  NOTE: printed values below are from ONE sample (ds[0]), NOT mean.")

    sample = ds[0]
    print(f"  Sample keys in y: {list(sample.y.keys())}")

    for lb in ds.labels:
        if lb not in sample.y:
            print(f"  [ERROR] label '{lb}' NOT in sample.y!")
        else:
            val = sample.y[lb]
            print(f"    {lb}: {val.item():.6f} (dtype={val.dtype})")

    # 测试 collate（这里只验证 key/shape）
    # ⚠️ 如果 ds 很大，这里不要把所有样本都 collate 进一个 batch，会占内存
    # 只抽一小批
    k = min(32, len(ds))
    samples = [ds[i] for i in range(k)]
    batch = collate_circuit_seq(samples)
    print(f"  Batch keys: {list(batch.keys())}")
    for lb in ds.labels:
        if lb in batch:
            print(f"    batch['{lb}'] shape: {tuple(batch[lb].shape)}")
        else:
            print(f"    [ERROR] batch missing label '{lb}'!")


@torch.no_grad()
def _print_label_statistics(tag: str, ds: CircuitSeqDataset, quantiles: bool = True) -> None:
    """
    打印该 split 中每个 label 的统计信息：
      mean / std / min / max / (optional) p1/p50/p99
    """
    if len(ds) == 0:
        print(f"\n[{tag}] EMPTY split - skip label statistics.")
        return

    print(f"\n[{tag}] Label statistics over {len(ds)} samples:")
    # 为了稳妥，不一次性把所有样本的 graph/seq 都加载进内存；
    # 这里只取 label（scalar），速度更快
    for lb in ds.labels:
        # 收集该 label 的所有值
        vals = []
        for i in range(len(ds)):
            vals.append(ds[i].y[lb].detach().cpu())
        v = torch.stack(vals, dim=0).to(torch.float32)  # (N,)

        mean_v = v.mean().item()
        std_v = v.std(unbiased=False).item()
        min_v = v.min().item()
        max_v = v.max().item()

        if quantiles and v.numel() >= 10:
            q1 = torch.quantile(v, 0.01).item()
            q50 = torch.quantile(v, 0.50).item()
            q99 = torch.quantile(v, 0.99).item()
            print(
                f"  {lb:>8s} | mean={mean_v:12.6f}  std={std_v:12.6f}  "
                f"min={min_v:12.6f}  max={max_v:12.6f}  "
                f"p01={q1:12.6f}  p50={q50:12.6f}  p99={q99:12.6f}"
            )
        else:
            print(
                f"  {lb:>8s} | mean={mean_v:12.6f}  std={std_v:12.6f}  "
                f"min={min_v:12.6f}  max={max_v:12.6f}"
            )


def test_split_with_labels(
    csv_path: str,
    circuit_dir: str,
    seq_dir: str,
    labels: Optional[List[SupportedLabel]] = None,
    mode: str = "within_circuit",
    preload_graphs: bool = True,
) -> None:
    """
    测试 split_dataset 在指定 labels 下是否正常工作。
    """
    labels_str = labels if labels else "all (default)"
    print("=" * 60)
    print(f"[TEST] mode={mode}, labels={labels_str}")
    print("=" * 60)

    # 1) 构建 Dataset
    ds = CircuitSeqDataset(
        csv_path=csv_path,
        circuit_dir=circuit_dir,
        seq_dir=seq_dir,
        use_header=True,
        check_paths=False,
        preload_graphs=preload_graphs,
        labels=labels,
    )

    print("\n[INFO] Full dataset")
    print("  samples:", len(ds))
    print("  unique circuits:", len(ds.circuits))
    print("  labels:", ds.labels)
    print("  df columns:", list(ds.df.columns))

    # 2) 分割
    train_ds, val_ds, test_ds = split_dataset(
        ds,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        mode=mode,
        seed=0,
        min_per_split_per_gid=0,
    )

    # 3) 验证 labels 传递
    print("\n[CHECK] Labels propagation to subsets:")
    print(f"  train_ds.labels: {train_ds.labels}")
    print(f"  val_ds.labels  : {val_ds.labels}")
    print(f"  test_ds.labels : {test_ds.labels}")

    assert train_ds.labels == ds.labels, "train_ds labels mismatch!"
    assert val_ds.labels == ds.labels, "val_ds labels mismatch!"
    assert test_ds.labels == ds.labels, "test_ds labels mismatch!"
    print("  [OK] All subsets have correct labels.")

    # 4) 打印统计：每个 split 每个电路样本数
    _print_split_circuit_stats("TRAIN", train_ds)
    _print_split_circuit_stats("VAL", val_ds)
    _print_split_circuit_stats("TEST", test_ds)

    # 4.5) 每个 split 的 label 统计信息（mean/min/max/std/quantiles）
    _print_label_statistics("TRAIN", train_ds, quantiles=True)
    _print_label_statistics("VAL", val_ds, quantiles=True)
    _print_label_statistics("TEST", test_ds, quantiles=True)

    # 5) 自检：gid 覆盖情况
    _print_overlap_check(train_ds, val_ds, test_ds)

    # 6) 测试 labels 在样本中是否正常工作
    _test_labels_in_sample(train_ds, "TRAIN")
    _test_labels_in_sample(val_ds, "VAL")
    _test_labels_in_sample(test_ds, "TEST")

    print("\n" + "=" * 60)
    print(f"[DONE] Test passed for mode={mode}, labels={labels_str}")
    print("=" * 60 + "\n")


def main():
    csv_path = "/home/yfdai/asap/data/data.csv"
    circuit_dir = "/home/yfdai/asap/data/aig"
    seq_dir = "/home/yfdai/asap/data/seq/"

    # Test 1: 默认所有 labels, within_circuit
    test_split_with_labels(
        csv_path=csv_path,
        circuit_dir=circuit_dir,
        seq_dir=seq_dir,
        labels=None,
        mode="within_circuit",
    )

    # Test 2: 只选择部分 labels
    test_split_with_labels(
        csv_path=csv_path,
        circuit_dir=circuit_dir,
        seq_dir=seq_dir,
        labels=["delay", "area"],
        mode="within_circuit",
    )

    # Test 3: by_circuit，单个 label
    test_split_with_labels(
        csv_path=csv_path,
        circuit_dir=circuit_dir,
        seq_dir=seq_dir,
        labels=["delay"],
        mode="by_circuit",
    )

    # Test 4: by_circuit，多个 labels
    test_split_with_labels(
        csv_path=csv_path,
        circuit_dir=circuit_dir,
        seq_dir=seq_dir,
        labels=["nd", "lev", "runtime"],
        mode="by_circuit",
    )

    print("\n" + "#" * 60)
    print("# ALL TESTS PASSED!")
    print("#" * 60)


if __name__ == "__main__":
    main()
