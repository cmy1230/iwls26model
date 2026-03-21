"""从一批序列文件中贪心选出覆盖最广的 N 个样本（Max-Min Diversity）。"""

from typing import List, Sequence
import numpy as np
from seq_preprocessing import load_seq


def encode_seq_as_freq(seq_path: str) -> np.ndarray:
    """将序列文件编码为指令频率向量 (D,)。"""
    tensor = load_seq(seq_path)          # (L, D)  float32
    L = tensor.shape[0]
    if L == 0:
        return np.zeros(tensor.shape[1], dtype=np.float32)
    counts = tensor.numpy().sum(axis=0)  # (D,)
    return (counts / L).astype(np.float32)


def greedy_maxmin_select(
    features: np.ndarray,
    n_select: int,
    seed: int = 42,
) -> List[int]:
    """
    Greedy Max-Min Diversity 选择。

    Parameters
    ----------
    features : (N, D) 特征矩阵
    n_select : 要选出的样本数
    seed     : 随机种子（仅在平局时用于打破对称）

    Returns
    -------
    选出样本的下标列表，长度 = min(n_select, N)
    """
    rng = np.random.RandomState(seed)
    N = features.shape[0]
    n_select = min(n_select, N)
    if n_select <= 0:
        return []

    mean = features.mean(axis=0)
    dists_to_mean = np.linalg.norm(features - mean, axis=1)
    first = int(np.argmax(dists_to_mean))

    selected: List[int] = [first]
    min_dists = np.linalg.norm(features - features[first], axis=1)
    min_dists[first] = -1.0

    for _ in range(n_select - 1):
        idx = int(np.argmax(min_dists))
        selected.append(idx)
        dist_to_new = np.linalg.norm(features - features[idx], axis=1)
        min_dists = np.minimum(min_dists, dist_to_new)
        min_dists[idx] = -1.0

    return selected


def select_diverse_samples(
    seq_paths: Sequence[str],
    n_select: int,
    seed: int = 42,
) -> List[int]:
    """
    从一批序列文件路径中选出覆盖最广的 n_select 个样本。

    Returns
    -------
    选出样本在 seq_paths 中的下标列表
    """
    freq_list = [encode_seq_as_freq(p) for p in seq_paths]
    features = np.stack(freq_list, axis=0)  # (N, 31)
    return greedy_maxmin_select(features, n_select, seed)


if __name__ == "__main__":
    import argparse, glob, json

    parser = argparse.ArgumentParser(description="Diversity-based sample selection")
    parser.add_argument("--seq_dir", type=str, required=True,
                        help="包含序列文件的目录")
    parser.add_argument("--pattern", type=str, default="*.txt",
                        help="文件匹配模式")
    parser.add_argument("-n", "--n_select", type=int, required=True,
                        help="要选出的样本数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = sorted(glob.glob(f"{args.seq_dir}/{args.pattern}"))
    if not paths:
        print("未找到匹配文件"); exit(1)

    indices = select_diverse_samples(paths, args.n_select, args.seed)
    print(f"从 {len(paths)} 个样本中选出 {len(indices)} 个:")
    for rank, i in enumerate(indices):
        print(f"  [{rank}] idx={i}  {paths[i]}")
