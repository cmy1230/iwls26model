# 序列读取的函数：输入是文件地址，输出是独热码编码好的序列

import os
from typing import List, Optional, Sequence, Dict
import numpy as np
import torch

# 你这篇代码里的命令表（D = 19）
DEFAULT_COMMANDS: List[str] = [
    "fraig;", "ifraig;", "dfraig;", "dc2;", "dch;", "dch -x;", "dc2 -l;", "dch -f;",
    "iresyn;", "balance;", "drf;", "drw;", "dc2 -b;", "rewrite -l;", "resub;", "resub -z;",
    "drwsat;", "irw;", "irws;", "rewrite -z;", "resub -l;", "resub -z -l;",
    "resub -K 8;" , "dch -x -f;", "dc2 -l -b;", "rewrite -z -l;", "refactor;", "refactor -z;", 
    "refactor -l;", "refactor -z -l;" , "&get -n; &dsdb; &put;"
]

def load_seq(
    seq_path: str,
    *,
    commands: Sequence[str] = DEFAULT_COMMANDS,
    start_line: int = 7,               # 从第8行开始 (0-based index=7)
    stop_token: str = "map;",
    unknown: str = "ignore",           # "ignore" | "error" | "zero"
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Read a log/script file and return sequence matrix for LSTM/Transformer.

    Returns:
        seq: torch.Tensor of shape (L, D), float32 by default.

    Parsing rule (consistent with your reference code):
      - Read lines from `start_line` (default line 8).
      - Collect commands until encountering `stop_token` (default "map;").
      - Each command is converted to one-hot over `commands`.

    unknown behavior:
      - "ignore": skip unknown command lines (L decreases)
      - "zero": keep a row of all-zeros for unknown command
      - "error": raise ValueError
    """
    if not os.path.exists(seq_path):
        raise FileNotFoundError(f"seq file not found: {seq_path}")

    cmd2idx: Dict[str, int] = {c: i for i, c in enumerate(commands)}
    D = len(commands)

    with open(seq_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # 防御：文件行数不足
    if len(lines) <= start_line:
        # 没有命令序列，返回空序列 (0, D)
        return torch.zeros((0, D), dtype=dtype)

    cmd_lines = []
    for raw in lines[start_line:]:
        s = raw.strip()
        if not s:
            continue
        if s == stop_token:
            break
        cmd_lines.append(s)

    # 生成 one-hot
    rows: List[np.ndarray] = []
    for s in cmd_lines:
        if s in cmd2idx:
            v = np.zeros((D,), dtype=np.float32)
            v[cmd2idx[s]] = 1.0
            rows.append(v)
        else:
            if unknown == "ignore":
                continue
            elif unknown == "zero":
                rows.append(np.zeros((D,), dtype=np.float32))
            else:
                raise ValueError(f"Unknown command line in {seq_path}: {s!r}")

    if len(rows) == 0:
        return torch.zeros((0, D), dtype=dtype)

    mat = np.stack(rows, axis=0)  # (L, D)
    return torch.tensor(mat, dtype=dtype)
