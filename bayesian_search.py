"""
BOiLS: Bayesian Optimisation for Logic Synthesis
基于 GP + Sub-Sequence String Kernel (SSK) 和 Hamming 信赖域搜索的 ABC 逻辑综合优化

核心算法 (替代原 Optuna/TPE):
  1. SSK 核函数: 通过公共子序列度量操作序列相似度, 替代 TPE 的独立分布建模
  2. 高斯过程 (GP): 基于 SSK 核构建代理模型, 提供预测均值 + 校准的不确定性估计
  3. Expected Improvement (EI): 采集函数, 平衡探索与利用
  4. Hamming 信赖域: 自适应半径约束, 在最优解邻域内局部搜索

参考: Grosnit et al., "BOiLS: Bayesian Optimisation for Logic Synthesis", DATE 2022

依赖: pip install numpy scipy

用法:
    python bayesian_search.py --abc_exe <abc路径> --input_file <aig/blif文件> [选项]

示例:
    # 基础优化 - AIG 输入
    python bayesian_search.py --abc_exe ./abc --input_file sin.aig

    # FPGA 映射优化 (LUT-K=6)
    python bayesian_search.py --abc_exe ./abc --input_file sin.blif --mapping FPGA --map_arg 6

    # ASIC 标准单元映射优化
    python bayesian_search.py --abc_exe ./abc --input_file sin.blif --mapping SCL --cell_lib mycells.genlib

    # 自定义搜索参数
    python bayesian_search.py --abc_exe ./abc --input_file sin.aig --n_trials 300 --ssk_order 3

    # 仅原子动作
    python bayesian_search.py --abc_exe ./abc --input_file sin.blif --no_macros --seq_len 30
"""

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import numpy as np
    from scipy.linalg import cho_factor, cho_solve
    from scipy.optimize import minimize as scipy_minimize
    from scipy.stats import norm as sp_norm
except ImportError:
    print("请先安装依赖:  pip install numpy scipy")
    sys.exit(1)

try:
    from numba import njit
    _SSK_BACKEND = "numba"
except ImportError:
    njit = None
    _SSK_BACKEND = "python"


# ============================================================================
#  动作空间定义
# ============================================================================

NOP = "__nop__"

ATOMIC_ACTIONS = [
    "fraig", "ifraig", "dfraig", "dc2", "dch", "dch -x", "dc2 -l", "dch -f",
    "iresyn", "balance", "drf", "drw", "dc2 -b", "rewrite -l", "resub", "resub -z",
    "drwsat", "irw", "irws", "rewrite -z", "resub -l", "resub -z -l",
    "resub -K 8", "dch -x -f", "dc2 -l -b", "rewrite -z -l",
    "refactor", "refactor -z", "refactor -l", "refactor -z -l",
    "&get -n; &dsdb; &put",
]

MACRO_ACTIONS = [
    # resyn
    "balance; rewrite; rewrite -z; balance; rewrite -z; balance",
    # resyn2
    "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance",
    # resyn2a
    "balance; rewrite; refactor; balance; rewrite -z; balance; refactor -z; rewrite -z; balance",
    # resyn3 (resub 主导)
    "balance; resub; resub -K 8; balance; resub -z; resub -z -K 8; balance; resub -K 8; balance",
    # dc2 深度压缩链
    "dc2; dc2 -l; dc2 -b; balance",
    # dch 组合验证链
    "dch; balance; dch -f; balance",
    "dch -x; balance; dch -x -f; balance",
    # 面积导向混合
    "rewrite; rewrite -z; refactor; refactor -z; resub; resub -z; balance",
    # 时序导向 (-l 保持层级)
    "balance; rewrite -l; refactor -l; resub -l; balance",
    # 大窗口 resub
    "resub; resub -K 8; resub -z; resub -z -K 8; balance",
    # &-space 深度优化
    "&get -n; &dsdb; &put; balance; rewrite -z; refactor -z; balance",
    # drw/drf 快速组合
    "drf; drw; drwsat; balance",
    # ifraig + dc2
    "ifraig; dc2; balance; dc2 -l; balance",
    "fraig; dc2; dch",
]

ACTIONS = ATOMIC_ACTIONS + MACRO_ACTIONS + [NOP]

try:
    from model_surrogate import ModelSurrogate
    _SURROGATE_AVAILABLE = True
except ImportError:
    ModelSurrogate = None
    _SURROGATE_AVAILABLE = False


def canonicalize_seq(seq, nop_idx):
    """NOP 尾部规范化: 保持非 NOP 元素相对顺序, 将所有 NOP 推到末尾。
    消除等价表示冗余, 例如 [A, NOP, B] 和 [A, B, NOP] 均规范化为 [A, B, NOP]。
    """
    non_nop = [v for v in seq if v != nop_idx]
    return non_nop + [nop_idx] * (len(seq) - len(non_nop))


# ============================================================================
#  ABC 交互
# ============================================================================

LIB_READ_CMD = {
    ".genlib": "read_genlib",
    ".lib":    "read_lib",
    ".super":  "read_super",
}


class ABCRunner:
    """通过 subprocess 调用 ABC 可执行文件"""

    def __init__(self, abc_exe: str, cell_libs: str = ""):
        self.abc_exe = str(Path(abc_exe).resolve())
        self.cell_libs = []

        if cell_libs:
            for lib_path in cell_libs.split(";"):
                lib_path = lib_path.strip()
                if not lib_path:
                    continue
                resolved = str(Path(lib_path).resolve())
                if not Path(resolved).exists():
                    raise FileNotFoundError(f"标准单元库文件不存在: {resolved}")
                self.cell_libs.append(resolved)

        if not Path(self.abc_exe).exists():
            raise FileNotFoundError(f"ABC 可执行文件不存在: {self.abc_exe}")

    @staticmethod
    def _needs_strash(input_file: str) -> bool:
        return Path(input_file).suffix.lower() != ".aig"

    def run_cmd(self, cmd_str: str, timeout: int = 300) -> str:
        try:
            result = subprocess.run(
                [self.abc_exe, "-c", cmd_str],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            print(f"[WARN] ABC 命令超时 ({timeout}s): {cmd_str[:80]}...")
            return ""
        except Exception as e:
            print(f"[ERROR] ABC 执行失败: {e}")
            return ""

    def get_stats(self, input_file: str) -> dict:
        strash = "strash; " if self._needs_strash(input_file) else ""
        cmd = f'read {input_file}; {strash}print_stats'
        output = self.run_cmd(cmd)
        return self._parse_print_stats(output)

    def _lib_read_cmds(self) -> list:
        cmds = []
        '''
        is_first = True
        for lib_path in self.cell_libs:
            suffix = Path(lib_path).suffix.lower()
            if is_first:
                read_cmd = LIB_READ_CMD.get(suffix, "read_lib -w")
                is_first = False
            else:
                read_cmd = LIB_READ_CMD.get(suffix, "read_lib -m -w")
            cmds.append(f'{read_cmd} {lib_path}')
        '''
        cmds.append("read_lib -w \"/home/yfdai/asap/data/lib/asap7sc7p5t_AO_RVT_FF_nldm_211120.lib\";\n")
        cmds.append("read_lib -m -w \"/home/yfdai/asap/data/lib/asap7sc7p5t_INVBUF_RVT_TT_nldm_220122.lib\";\n")
        cmds.append("read_lib -m -w \"/home/yfdai/asap/data/lib/asap7sc7p5t_OA_RVT_TT_nldm_211120.lib\";\n")
        cmds.append("read_lib -m -w \"/home/yfdai/asap/data/lib/asap7sc7p5t_SIMPLE_RVT_TT_nldm_211120.lib\";\n")
        return cmds

    def run_sequence_and_stats(self, input_file: str, action_seq: list,
                                mapping: str = "", map_arg: str = "",
                                map_tail: str = "") -> dict:
        parts = self._lib_read_cmds() + [f'read {input_file}']

        if self._needs_strash(input_file):
            parts.append('strash')

        if action_seq:
            parts.append("; ".join(action_seq))

        if mapping == "FPGA":
            if map_tail:
                parts.append(map_tail)
            else:
                k = map_arg if map_arg else "6"
                parts.append(f'if -K {k}')
            parts.append('print_stats')
        elif mapping == "SCL":
            if map_tail:
                parts.append(map_tail)
            else:
                parts.append(f'map')
                #parts.append('topo')
            parts.append('print_stats')
        else:
            parts.append('print_stats')

        cmd = "; ".join(parts)
        output = self.run_cmd(cmd)
        return self._parse_print_stats(output)

    def run_fast_stats(self, input_file: str, action_seq: list) -> dict:
        """
        快速评估：执行优化序列但跳过技术映射，只返回 AIG 统计。
        用于多保真度模块（--multifidelity）的低保真路径。
        """
        parts = self._lib_read_cmds() + [f'read {input_file}']
        if self._needs_strash(input_file):
            parts.append('strash')
        if action_seq:
            parts.append("; ".join(action_seq))
        parts.append('print_stats')
        cmd = "; ".join(parts)
        output = self.run_cmd(cmd)
        return self._parse_print_stats(output)

    @staticmethod
    def _parse_print_stats(output: str) -> dict:
        stats = {"nodes": -1, "levels": -1, "area": -1.0, "delay": -1.0, "edges": -1}

        if not output:
            return stats

        lines = output.strip().split("\n")
        stat_line = ""
        for line in reversed(lines):
            if "i/o" in line or "nd" in line or "and" in line:
                stat_line = line
                break

        if not stat_line:
            return stats

        m_and = re.search(r'and\s*=\s*(\d+)', stat_line)
        m_nd = re.search(r'nd\s*=\s*(\d+)', stat_line)
        m_lev = re.search(r'lev\s*=\s*(\d+)', stat_line)
        m_edge = re.search(r'edge\s*=\s*(\d+)', stat_line)
        m_area = re.search(r'area\s*=\s*([\d.]+)', stat_line)
        m_delay = re.search(r'delay\s*=\s*([\d.]+)', stat_line)

        if m_and:
            stats["nodes"] = int(m_and.group(1))
        if m_nd:
            stats["nodes"] = int(m_nd.group(1))
        if m_lev:
            stats["levels"] = int(m_lev.group(1))
        if m_edge:
            stats["edges"] = int(m_edge.group(1))
        if m_area:
            stats["area"] = float(m_area.group(1))
        if m_delay:
            stats["delay"] = float(m_delay.group(1))

        return stats


# ============================================================================
#  Pareto 前沿跟踪 (模块F)
#
#  记录搜索过程中所有 (metric_a, metric_b) 非支配点。
#  有映射时用 (area_ratio, delay_ratio)，无映射时用 (nodes_ratio, levels_ratio)。
# ============================================================================

class ParetoFrontier:
    """维护 QoR 双目标 Pareto 最优前沿"""

    def __init__(self):
        self.points = []
        # 每条记录格式: (metric_a, metric_b, seq_indices_list, stats_dict)

    def update(self, metric_a: float, metric_b: float,
               seq_indices: list, stats: dict):
        """
        尝试将新点加入前沿。
        metric_a, metric_b 均为越小越好（ratio值）。
        """
        if metric_a <= 0 or metric_b <= 0:
            return
        # 被任意现有点支配则丢弃
        for p in self.points:
            if p[0] <= metric_a and p[1] <= metric_b:
                return
        # 移除被新点支配的旧点
        self.points = [
            p for p in self.points
            if not (metric_a <= p[0] and metric_b <= p[1])
        ]
        self.points.append((metric_a, metric_b,
                            list(seq_indices), dict(stats)))

    def sorted_points(self):
        return sorted(self.points, key=lambda x: x[0])

    def summary_str(self):
        pts = self.sorted_points()
        if not pts:
            return "  Pareto 前沿: 无有效点（可能无映射数据）"
        lines = [f"  Pareto 前沿共 {len(pts)} 个非支配点:"]
        for a, b, _, _ in pts:
            lines.append(f"    metric_a={a:.4f}  metric_b={b:.4f}")
        return "\n".join(lines)

    def to_dict_list(self):
        return [
            {"metric_a": p[0], "metric_b": p[1], "stats": p[3]}
            for p in self.sorted_points()
        ]


# ============================================================================
#  Sub-Sequence String Kernel (SSK)
#
#  k_p(s, t) = Σ_{u ∈ Alg^p} c_u(s) · c_u(t)
#  其中 c_u(s) = θ_m^|u| · Σ_i θ_g^{gap(u,i)} · I(s_i = u)
#
#  θ_m: 匹配衰减 — 控制长子序列的权重
#  θ_g: 间隔衰减 — 惩罚不连续的子序列匹配
#  使用 O(p · |s| · |t|) 的 DP 算法 (Lodhi et al., 2002)
#  当 Numba 可用时自动 JIT 编译加速 (~50-100x)
# ============================================================================

# ---- Pure Python fallback ----

def _raw_ssk_dp_py(s, t, order, tm2, tg):
    """Pure Python: DP 计算原始 (未归一化) SSK 值"""
    ls, lt = len(s), len(t)
    if min(ls, lt) < order:
        return 0.0

    w = lt + 1
    Kp = [1.0] * ((ls + 1) * w)
    result = 0.0

    for p in range(1, order + 1):
        nKp = [0.0] * ((ls + 1) * w)
        last = (p == order)
        for i in range(1, ls + 1):
            si = s[i - 1]
            acc = 0.0
            ri = i * w
            rim1 = (i - 1) * w
            for j in range(1, lt + 1):
                if si == t[j - 1]:
                    acc = tg * acc + tm2 * Kp[rim1 + j - 1]
                    if last:
                        result += acc
                else:
                    acc *= tg
                nKp[ri + j] = tg * nKp[rim1 + j] + acc
        Kp = nKp

    return result


# ---- Numba JIT 加速路径 ----

if _SSK_BACKEND == "numba":

    @njit(cache=True)
    def _ssk_core_jit(s, t, order, tm2, tg):
        """Numba JIT: SSK 核心 DP (接收 numpy int64 数组)"""
        ls = len(s)
        lt = len(t)
        if min(ls, lt) < order:
            return 0.0
        w = lt + 1
        sz = (ls + 1) * w
        Kp = np.ones(sz)
        result = 0.0
        for p in range(1, order + 1):
            nKp = np.zeros(sz)
            last = (p == order)
            for i in range(1, ls + 1):
                si = s[i - 1]
                acc = 0.0
                ri = i * w
                rim1 = (i - 1) * w
                for j in range(1, lt + 1):
                    if si == t[j - 1]:
                        acc = tg * acc + tm2 * Kp[rim1 + j - 1]
                        if last:
                            result += acc
                    else:
                        acc *= tg
                    nKp[ri + j] = tg * nKp[rim1 + j] + acc
            Kp = nKp
        return result

    @njit(cache=True)
    def _ssk_gram_raw_jit(seqs, order, tm2, tg):
        """Numba JIT: 批量计算原始 SSK Gram 矩阵 (避免 N² 次 Python 调用开销)"""
        n = seqs.shape[0]
        K = np.empty((n, n))
        for i in range(n):
            K[i, i] = _ssk_core_jit(seqs[i], seqs[i], order, tm2, tg)
            for j in range(i + 1, n):
                v = _ssk_core_jit(seqs[i], seqs[j], order, tm2, tg)
                K[i, j] = v
                K[j, i] = v
        return K

    @njit(cache=True)
    def _ssk_kvec_raw_jit(train, x, order, tm2, tg):
        """Numba JIT: 批量计算 x 与训练集的原始 SSK 向量"""
        n = train.shape[0]
        kv = np.empty(n)
        for i in range(n):
            kv[i] = _ssk_core_jit(train[i], x, order, tm2, tg)
        return kv

    @njit(cache=True)
    def _ssk_gram_raw_vl_jit(seqs, lengths, order, tm2, tg):
        """Numba JIT: 变长序列的批量 Gram 矩阵 (按 lengths 截取有效部分)"""
        n = seqs.shape[0]
        K = np.empty((n, n))
        for i in range(n):
            si = seqs[i, :lengths[i]]
            K[i, i] = _ssk_core_jit(si, si, order, tm2, tg)
            for j in range(i + 1, n):
                sj = seqs[j, :lengths[j]]
                v = _ssk_core_jit(si, sj, order, tm2, tg)
                K[i, j] = v
                K[j, i] = v
        return K

    @njit(cache=True)
    def _ssk_kvec_raw_vl_jit(train, train_lens, x, x_len, order, tm2, tg):
        """Numba JIT: 变长序列的批量核向量"""
        n = train.shape[0]
        x_eff = x[:x_len]
        kv = np.empty(n)
        for i in range(n):
            kv[i] = _ssk_core_jit(train[i, :train_lens[i]], x_eff, order, tm2, tg)
        return kv

    # 预热 JIT: 首次编译约 1-2s, 后续从磁盘缓存加载 (__pycache__)
    _ssk_core_jit(np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64),
                  1, 0.5, 0.5)
    _ssk_gram_raw_vl_jit(np.zeros((2, 2), dtype=np.int64),
                         np.array([2, 2], dtype=np.int64), 1, 0.5, 0.5)
    _ssk_kvec_raw_vl_jit(np.zeros((2, 2), dtype=np.int64),
                         np.array([2, 2], dtype=np.int64),
                         np.zeros(2, dtype=np.int64), np.int64(2),
                         1, 0.5, 0.5)

    def _raw_ssk_dp(s, t, order, tm2, tg):
        """SSK DP (Numba 加速, 自动转换输入)"""
        return float(_ssk_core_jit(
            np.asarray(s, dtype=np.int64),
            np.asarray(t, dtype=np.int64),
            order, tm2, tg))

else:
    _raw_ssk_dp = _raw_ssk_dp_py


class SSKKernel:
    """面向逻辑综合操作序列的 Sub-Sequence String Kernel (NOP 感知)

    归一化核:  k̂(s,t) = σ² · k_raw(s,t) / √(k_raw(s,s) · k_raw(t,t))
    对角项 (含噪声):  k̂(s,s) = σ² + σ_n²

    NOP 处理 (方案 B):
      - 输入序列应已经过 canonicalize_seq 规范化 (NOP 在末尾)
      - 核计算前自动剥离 NOP, 仅对有效操作子序列计算 SSK
      - 消除 [A, NOP, B] ≡ [A, B, NOP] 等等价表示的核值差异

    当 Numba 可用时, gram_matrix / kernel_vector 使用批量 JIT 路径,
    通过 lengths 数组实现变长序列的高效批量计算。
    """

    def __init__(self, subseq_order=2, theta_m=0.8, theta_g=0.8,
                 signal_var=1.0, noise_var=1e-2, nop_idx=None,
                 circuit_weight=0.3, circuit_sigma=0.5):
        self.order       = subseq_order
        self.theta_m     = theta_m
        self.theta_g     = theta_g
        self.signal_var  = signal_var
        self.noise_var   = noise_var
        self.nop_idx     = nop_idx
        # CC-SSK 参数（模块C）
        # circuit_weight=0: 退回纯 SSK；>0: 混合电路相似度
        self.circuit_weight = float(np.clip(circuit_weight, 0.0, 1.0))
        self.circuit_sigma  = max(float(circuit_sigma), 1e-6)
        # 由 BOiLSOptimizer.run() 在每次 gp.fit() 前注入，shape (n, d)
        # 注意: gram_matrix/kernel_vector 本身不使用此字段，
        #       由 GaussianProcessSSK.fit() 在组装 K 后调用 apply_cc_ssk()
        self._circuit_feats = None
        self._cache         = {}
        self._diag_cache    = {}
        self._jit_train_2d   = None
        self._jit_train_diag = None
        self._jit_train_lens = None

    def _strip(self, s):
        """剥离 NOP, 返回仅含有效操作的 tuple (用于缓存键和 Python 路径计算)"""
        if self.nop_idx is None:
            return tuple(s)
        return tuple(v for v in s if v != self.nop_idx)

    def _effective_len(self, s):
        """对于已规范化的序列 (NOP 在末尾), 返回有效操作长度"""
        if self.nop_idx is None:
            return len(s)
        for i, v in enumerate(s):
            if v == self.nop_idx:
                return i
        return len(s)

    def _raw(self, s, t):
        s_eff = self._strip(s)
        t_eff = self._strip(t)
        key = (s_eff, t_eff)
        v = self._cache.get(key)
        if v is not None:
            return v
        sym = (t_eff, s_eff)
        v = self._cache.get(sym)
        if v is not None:
            self._cache[key] = v
            return v
        v = _raw_ssk_dp(list(s_eff), list(t_eff), self.order,
                        self.theta_m ** 2, self.theta_g)
        self._cache[key] = v
        return v

    def _raw_self(self, s):
        key = self._strip(s)
        v = self._diag_cache.get(key)
        if v is not None:
            return v
        v = self._raw(s, s)
        self._diag_cache[key] = v
        return v

    def normalized(self, s, t):
        rs = self._raw_self(s)
        rt = self._raw_self(t)
        if rs <= 0 or rt <= 0:
            return 0.0
        return self._raw(s, t) / math.sqrt(rs * rt)

    def __call__(self, s, t, noise=False):
        v = self.signal_var * self.normalized(s, t)
        if noise and self._strip(s) == self._strip(t):
            v += self.noise_var
        return v

    def gram_matrix(self, seqs):
        n = len(seqs)
        if _SSK_BACKEND == "numba":
            max_len = max((len(s) for s in seqs), default=0)
            seqs_2d = np.zeros((n, max_len), dtype=np.int64)
            for i, s in enumerate(seqs):
                seqs_2d[i, :len(s)] = s
            lengths = np.array([self._effective_len(s) for s in seqs],
                               dtype=np.int64)
            tm2 = self.theta_m ** 2
            raw_K = _ssk_gram_raw_vl_jit(seqs_2d, lengths,
                                         self.order, tm2, self.theta_g)
            diag = np.diag(raw_K).copy()
            denom = np.sqrt(np.maximum(np.outer(diag, diag), 1e-300))
            K = self.signal_var * raw_K / denom
            np.fill_diagonal(K, self.signal_var + self.noise_var)
            self._jit_train_2d = seqs_2d
            self._jit_train_diag = diag
            self._jit_train_lens = lengths
            return K
        K = np.zeros((n, n))
        raw_diags = [self._raw_self(s) for s in seqs]
        for i in range(n):
            K[i, i] = self.signal_var + self.noise_var
            for j in range(i + 1, n):
                rij = self._raw(seqs[i], seqs[j])
                d = raw_diags[i] * raw_diags[j]
                v = self.signal_var * rij / math.sqrt(d) if d > 0 else 0.0
                K[i, j] = K[j, i] = v
        return K

    def kernel_vector(self, seqs, x):
        if _SSK_BACKEND == "numba" and self._jit_train_2d is not None:
            x_arr = np.asarray(x, dtype=np.int64)
            x_len = np.int64(self._effective_len(x))
            tm2 = self.theta_m ** 2
            raw_kv = _ssk_kvec_raw_vl_jit(
                self._jit_train_2d, self._jit_train_lens,
                x_arr, x_len, self.order, tm2, self.theta_g)
            x_eff = x_arr[:x_len]
            rx = float(_ssk_core_jit(
                x_eff, x_eff, self.order, tm2, self.theta_g))
            denom = np.sqrt(np.maximum(self._jit_train_diag * rx, 1e-300))
            return self.signal_var * raw_kv / denom
        rx = self._raw_self(x)
        kv = np.zeros(len(seqs))
        for i, s in enumerate(seqs):
            ri = self._raw_self(s)
            rij = self._raw(s, x)
            d = ri * rx
            kv[i] = self.signal_var * rij / math.sqrt(d) if d > 0 else 0.0
        return kv

    def _extend_jit_cache(self, new_seq):
        """增量追加一个序列到 JIT 批量缓存 (Numba 路径用)"""
        if self._jit_train_2d is None:
            return
        new_len = len(new_seq)
        cur_width = self._jit_train_2d.shape[1]
        if new_len <= cur_width:
            # 新序列较短：zero-pad 到现有宽度
            new_arr = np.zeros((1, cur_width), dtype=np.int64)
            new_arr[0, :new_len] = new_seq
            self._jit_train_2d = np.vstack([self._jit_train_2d, new_arr])
        else:
            # 新序列较长：整体扩列，旧行右侧补零
            pad = np.zeros((self._jit_train_2d.shape[0], new_len - cur_width), dtype=np.int64)
            expanded = np.hstack([self._jit_train_2d, pad])
            new_arr = np.asarray(new_seq, dtype=np.int64).reshape(1, -1)
            self._jit_train_2d = np.vstack([expanded, new_arr])
        new_eff_len = np.int64(self._effective_len(new_seq))
        self._jit_train_lens = np.append(self._jit_train_lens, new_eff_len)
        x_eff = np.asarray(new_seq, dtype=np.int64)[:new_eff_len]
        tm2 = self.theta_m ** 2
        if _SSK_BACKEND == "numba":
            rx = float(_ssk_core_jit(x_eff, x_eff, self.order, tm2, self.theta_g))
        else:
            rx = _raw_ssk_dp(list(x_eff), list(x_eff), self.order, tm2, self.theta_g)
        self._jit_train_diag = np.append(self._jit_train_diag, rx)

    def clear_cache(self):
        self._cache.clear()
        self._diag_cache.clear()
        self._jit_train_2d = None
        self._jit_train_diag = None
        self._jit_train_lens = None

    def set_params(self, theta_m, theta_g, signal_var=None):
        self.theta_m = float(np.clip(theta_m, 0.01, 0.99))
        self.theta_g = float(np.clip(theta_g, 0.01, 0.99))
        if signal_var is not None:
            self.signal_var = max(float(signal_var), 0.01)
        self.clear_cache()

    def apply_cc_ssk(self, K: np.ndarray) -> np.ndarray:
        """
        对已计算好的 SSK Gram 矩阵 K 施加电路状态相似度权重。

        CC-SSK 公式:
          k_CC(i,j) = k_SSK(i,j) * [ (1-w) + w * exp(-||feat_i - feat_j||^2 / 2σ^2) ]

        因为 C[i,i] = exp(0) = 1，所以 W[i,i] = 1.0，对角线自动保持不变。

        注意: 此方法不修改 K 本身，返回新矩阵。
        """
        n = K.shape[0]
        if (self.circuit_weight == 0.0
                or self._circuit_feats is None
                or len(self._circuit_feats) != n):
            return K  # 无特征或权重为0，直接返回原矩阵

        feats = np.asarray(self._circuit_feats, dtype=np.float64)  # n×d
        # 向量化计算 n×n 距离平方矩阵
        diff  = feats[:, np.newaxis, :] - feats[np.newaxis, :, :]  # n×n×d
        dist2 = np.sum(diff ** 2, axis=-1)                         # n×n
        C     = np.exp(-dist2 / (2.0 * self.circuit_sigma ** 2))   # n×n, [0,1]
        W     = (1.0 - self.circuit_weight) + self.circuit_weight * C  # n×n, [(1-w),1]
        # W[i,i] = 1.0 恒成立，对角线不变
        return K * W

    def apply_cc_ssk_vector(self, kv: np.ndarray, x_feat=None) -> np.ndarray:
        """
        对预测时的核向量 k(x*, X) 施加 CC 权重。

        x_feat 为候选 x* 的结构特征向量；若为 None 则退回训练特征均值（fallback）。
        """
        n = len(kv)
        if (self.circuit_weight == 0.0
                or self._circuit_feats is None
                or len(self._circuit_feats) != n):
            return kv

        feats = np.asarray(self._circuit_feats, dtype=np.float64)
        if x_feat is None:
            x_feat = feats.mean(axis=0)
        diff = feats - np.asarray(x_feat, dtype=np.float64)[np.newaxis, :]
        dist2 = np.sum(diff ** 2, axis=-1)
        c = np.exp(-dist2 / (2.0 * self.circuit_sigma ** 2))
        w = (1.0 - self.circuit_weight) + self.circuit_weight * c
        return kv * w


# ============================================================================
#  Gaussian Process (SSK 核)
#
#  后验:  f|D ~ N(μ_post, Σ_post)
#    μ_post  = K(X*, X) K(X, X)^{-1} y
#    Σ_post  = K(X*, X*) − K(X*, X) K(X, X)^{-1} K(X, X*)
#  超参数通过最小化负对数边际似然 (Eq. 4) 学习
# ============================================================================

class GaussianProcessSSK:
    """基于 SSK 核的高斯过程回归 (支持增量 Gram 矩阵)"""

    def __init__(self, kernel: SSKKernel):
        self.kernel = kernel
        self.X = []
        self.y = np.array([])
        self.K = None
        self.L_cho = None
        self.alpha = None
        self._K_cache = None
        self._hp_stamp = None

    def _hp_key(self):
        k = self.kernel
        return (k.theta_m, k.theta_g, k.signal_var, k.noise_var)

    def fit(self, X, y):
        self.X = [list(x) for x in X]
        self.y = np.asarray(y, dtype=float)
        n = len(X)

        hp       = self._hp_key()
        n_cached = self._K_cache.shape[0] if self._K_cache is not None else 0

        # --- 组装原始 SSK Gram 矩阵（不含 CC 权重）---
        if self._hp_stamp == hp and 0 < n_cached < n:
            # 增量路径：复用旧缓存，只计算新行列
            K = np.zeros((n, n))
            K[:n_cached, :n_cached] = self._K_cache
            for i in range(n_cached, n):
                kv = self.kernel.kernel_vector(self.X[:i], self.X[i])
                K[:i, i] = kv
                K[i, :i] = kv
                K[i, i]  = self.kernel.signal_var + self.kernel.noise_var
                self.kernel._extend_jit_cache(self.X[i])
        elif self._hp_stamp == hp and n_cached == n:
            K = self._K_cache.copy()
        else:
            K = self.kernel.gram_matrix(self.X)

        # 缓存原始 SSK K（不含 CC 权重），供下次增量复用
        self._K_cache = K.copy()
        self._hp_stamp = hp

        # --- 应用 CC-SSK 权重（模块C）---
        # apply_cc_ssk 读取 kernel._circuit_feats，若为 None 则原样返回
        self.K = self.kernel.apply_cc_ssk(K)

        self._factor()

    def _factor(self):
        n = len(self.y)
        for jitter in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1):
            try:
                K_work = self.K.copy()
                K_work[np.diag_indices(n)] += jitter
                self.L_cho, self._lo = cho_factor(K_work, lower=True)
                self.alpha = cho_solve((self.L_cho, self._lo), self.y)
                if jitter > 1e-4:
                    print(f"  [GP] Cholesky 收敛 (jitter={jitter:.0e})")
                return
            except np.linalg.LinAlgError:
                continue
        eigvals, eigvecs = np.linalg.eigh(self.K)
        eigvals = np.maximum(eigvals, 1e-4)
        K_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        self.L_cho, self._lo = cho_factor(K_fixed, lower=True)
        self.alpha = cho_solve((self.L_cho, self._lo), self.y)
        print("  [GP] 特征值修正强制正定")

    def predict(self, X_new, X_new_feats=None):
        mu  = np.zeros(len(X_new))
        var = np.zeros(len(X_new))
        for i, x in enumerate(X_new):
            kv = self.kernel.kernel_vector(self.X, x)
            x_feat = X_new_feats[i] if X_new_feats is not None else None
            kv = self.kernel.apply_cc_ssk_vector(kv, x_feat=x_feat)
            mu[i]  = kv @ self.alpha
            v      = cho_solve((self.L_cho, self._lo), kv)
            var[i] = max(self.kernel.signal_var - kv @ v, 1e-10)
        return mu, var

    def neg_log_ml(self):
        n = len(self.y)
        log_det = 2.0 * np.sum(np.log(np.maximum(np.diag(self.L_cho), 1e-300)))
        return 0.5 * (log_det + float(self.y @ self.alpha) + n * np.log(2 * np.pi))

    def optimize_hp(self, n_restarts=2, max_iter=5):
        """通过边际似然优化核超参数 (θ_m, θ_g, σ²_f)"""
        best_nlml = self.neg_log_ml()
        best_p = (self.kernel.theta_m, self.kernel.theta_g, self.kernel.signal_var)
        X_save = [list(x) for x in self.X]
        y_save = self.y.copy()

        rng = np.random.default_rng()

        def obj(params):
            self.kernel.set_params(params[0], params[1], params[2])
            self.fit(X_save, y_save)
            return self.neg_log_ml()

        for _ in range(n_restarts):
            x0 = [rng.uniform(0.60, 0.98), rng.uniform(0.1, 0.95),
                   rng.uniform(0.5, 2.0)]
            try:
                r = scipy_minimize(
                    obj, x0,
                    bounds=[(0.60, 0.98), (0.05, 0.95), (0.05, 5.0)],
                    method="L-BFGS-B",
                    options={"maxiter": max_iter, "ftol": 1e-3},
                )
                if r.fun < best_nlml:
                    best_nlml = r.fun
                    best_p = (float(r.x[0]), float(r.x[1]), float(r.x[2]))
            except Exception:
                pass

        self.kernel.set_params(*best_p)
        self.fit(X_save, y_save)


# ============================================================================
#  Expected Improvement 采集函数
#
#  EI(x) = (μ(x) − f⁺) Φ(z) + σ(x) φ(z),  z = (μ(x) − f⁺) / σ(x)
#  其中 f⁺ = max observed value, Φ/φ 为标准正态 CDF/PDF
# ============================================================================

def expected_improvement(mu, var, y_best):
    sigma = np.sqrt(np.maximum(var, 1e-20))
    improve = mu - y_best
    z = np.where(sigma > 1e-10, improve / sigma, 0.0)
    ei = np.where(
        sigma > 1e-10,
        improve * sp_norm.cdf(z) + sigma * sp_norm.pdf(z),
        np.maximum(improve, 0.0),
    )
    return ei


# ============================================================================
#  Trust Region — 基于 Hamming 距离的自适应信赖域
#
#  TR(seq_best, ρ) = { seq ∈ Alg^K : Hamming(seq_best, seq) ≤ ρ }
#  半径调度:
#    - 连续 τ_succ 次改进 → ρ += 1  (扩大探索)
#    - 连续 τ_fail 次无改进 → ρ -= 1  (收缩聚焦)
#    - ρ → 0 时触发重启
# ============================================================================

class TrustRegion:

    def __init__(self, seq_len, n_actions, init_radius=None,
                 expand_thresh=3, shrink_thresh=20):
        self.seq_len = seq_len
        self.n_actions = n_actions
        self.radius = init_radius if init_radius is not None else seq_len
        self.expand_thresh = expand_thresh
        self.shrink_thresh = shrink_thresh
        self._succ = 0
        self._fail = 0
        self.restarts = 0

    def update(self, improved: bool):
        if improved:
            self._succ += 1
            self._fail = 0
            if self._succ >= self.expand_thresh:
                self.radius = min(self.radius + 1, self.seq_len)
                self._succ = 0
        else:
            self._fail += 1
            self._succ = 0
            if self._fail >= self.shrink_thresh:
                self.radius = max(self.radius - 1, 0)
                self._fail = 0

    @property
    def dead(self):
        return self.radius <= 0

    def restart(self):
        self.radius = self.seq_len
        self._succ = self._fail = 0
        self.restarts += 1

    def sample(self, center, n, rng):
        """在信赖域内采样: 随机选择 ≤ ρ 个位置并替换为不同的动作"""
        out = []
        rho = max(1, self.radius)
        center_len = len(center)
        if center_len <= 0:
            return out
        for _ in range(n):
            nc = int(rng.integers(1, rho + 1))
            pos = rng.choice(center_len, size=min(nc, center_len), replace=False)
            s = list(center)
            for p in pos:
                cands = [a for a in range(self.n_actions) if a != center[p]]
                if cands:
                    s[p] = int(rng.choice(cands))
            out.append(s)
        return out


# ============================================================================
#  渐进序列长度扩展 PLE (模块B) —— 独创
#
#  从短序列 (init_len) 开始搜索，TR 每收敛 patience 次则扩展 +step，
#  保留最优前缀，只在新增后缀空间探索。
#  直觉: 优秀的长序列通常是优秀的短序列加少量额外变换。
# ============================================================================

class ProgressiveLengthExpansion:

    def __init__(self, init_len: int, max_len: int,
                 step: int = 4, patience: int = 2):
        """
        init_len : 初始序列长度
        max_len  : 最大长度（即 --seq_len 参数）
        step     : 每次扩展步长
        patience : TR 需重启多少次才触发一次扩展
        """
        self.current_len             = max(init_len, 3)
        self.max_len                 = max_len
        self.step                    = step
        self.patience                = patience
        self._restarts_since_expand  = 0
        self.n_expansions            = 0
        self.expansion_log           = []  # [(id, old_len, new_len, best_cost)]

    @property
    def at_max(self) -> bool:
        return self.current_len >= self.max_len

    def on_tr_restart(self, best_seq: list, best_cost: float,
                      n_actions: int, rng) -> tuple:
        """
        TR dead 时调用。
        返回 (new_center, expanded):
          new_center : 扩展后的新起始序列；None 表示正常随机重启
          expanded   : 是否发生了长度扩展
        """
        self._restarts_since_expand += 1

        if not self.at_max and self._restarts_since_expand >= self.patience:
            old_len          = self.current_len
            self.current_len = min(self.current_len + self.step, self.max_len)
            self._restarts_since_expand = 0
            self.n_expansions += 1
            self.expansion_log.append(
                (self.n_expansions, old_len, self.current_len, best_cost))

            # 保留最优前缀（截到旧长度），随机补充后缀
            prefix     = list(best_seq)[:old_len]
            suffix_len = self.current_len - len(prefix)
            suffix     = list(rng.integers(0, n_actions, size=suffix_len))
            new_center = prefix + suffix

            print(f"  [PLE] 扩展 #{self.n_expansions}: "
                  f"seq_len {old_len} → {self.current_len}  "
                  f"best_cost={best_cost:.6f}")
            return new_center, True

        return None, False


# ============================================================================
#  电路评估器
# ============================================================================

class SynthesisEvaluator:
    """将动作索引序列转换为 ABC 命令并评估 QoR"""

    def __init__(self, abc_runner: ABCRunner, input_file: str, actions: list,
                 seq_len: int, optimize: str,
                 mapping: str, map_arg: str, map_tail: str,
                 multifidelity: bool = False):
        self.abc = abc_runner
        self.input_file = str(Path(input_file).resolve())
        self.actions = actions
        self.seq_len = seq_len
        self.optimize = optimize
        self.mapping = mapping
        self.map_arg = map_arg
        self.map_tail = map_tail

        if not Path(self.input_file).exists():
            raise FileNotFoundError(f"输入文件不存在: {self.input_file}")

        self.init_stats = self._get_init_stats()
        if self.init_stats["nodes"] <= 0:
            raise RuntimeError(f"无法获取初始电路统计信息，请检查 ABC 路径和输入文件。\n"
                               f"  abc_exe: {self.abc.abc_exe}\n  input_file: {self.input_file}")

        real_actions = [a for a in self.actions if a != NOP]
        n_atomic = sum(1 for a in real_actions if "; " not in a)
        n_macro = len(real_actions) - n_atomic
        print(f"\n{'='*60}")
        print(f"初始电路统计: {self.init_stats}")
        print(f"动作空间: {n_atomic} 原子 + {n_macro} 宏 + NOP = {len(self.actions)} 个选择")
        if n_macro > 0:
            for a in real_actions:
                if "; " in a:
                    print(f"  宏: {a}")
        print(f"固定参数维度: {self.seq_len} (NOP 自动跳过，实际长度自适应)")
        print(f"优化目标: {self.optimize}")
        print(f"SSK 加速: {_SSK_BACKEND}" +
              (" (pip install numba 可获得 ~50-100x 加速)" if _SSK_BACKEND == "python" else ""))
        if self.mapping:
            print(f"映射方式: {self.mapping} (参数: {self.map_arg or '默认'})")
        print(f"{'='*60}\n")

        self.best_cost = float("inf")
        self.best_seq = []
        self.best_stats = {}
        self._last_stats = {}
        self.eval_count = 0

        # 模块E: 多保真度评估（默认关闭，--multifidelity 开启）
        self.multifidelity       = multifidelity
        self._ei_hint            = 0.0   # 由 BOiLSOptimizer 在调用前注入
        self._ei_threshold       = 0.0   # 同上
        self._fast_eval_count    = 0
        self._full_eval_count    = 0

        # 模块C: 电路特征记录（与 BOiLSOptimizer.X 一一对应）
        # 每次 __call__ 无论成功与否都追加，保证长度与 X 同步
        self._circuit_features   = []    # list of np.ndarray shape (3,)

        # 模块F: Pareto 前沿跟踪
        self.pareto              = ParetoFrontier()

    def _get_init_stats(self) -> dict:
        if self.mapping:
            return self.abc.run_sequence_and_stats(
                self.input_file, [], self.mapping, self.map_arg, self.map_tail
            )
        return self.abc.get_stats(self.input_file)

    def _compute_cost(self, stats: dict) -> float:
        init = self.init_stats
        if self.mapping and stats.get("area", -1) > 0 and stats.get("delay", -1) > 0:
            a_ratio = stats["area"] / max(init["area"], 1e-10)
            d_ratio = stats["delay"] / max(init["delay"], 1e-10)
        else:
            a_ratio = stats["nodes"] / max(init["nodes"], 1)
            d_ratio = stats["levels"] / max(init["levels"], 1)

        if self.optimize == "area":
            return a_ratio
        elif self.optimize == "delay":
            return d_ratio
        else:
            return a_ratio * d_ratio

    def indices_to_strs(self, indices):
        return [self.actions[i] for i in indices if self.actions[i] != NOP]

    def __call__(self, indices) -> float:
        action_strs = self.indices_to_strs(indices)
        if not action_strs:
            # 全NOP序列：仍需追加特征以保持索引同步
            self._circuit_features.append(
                np.array([1.0, 1.0, 1.0], dtype=np.float64))
            return 10.0

        # ---- 模块E: 多保真度路径选择 ----
        # 规则优先级（从高到低）:
        #   1. multifidelity=False（默认）→ 永远完整评估
        #   2. 每 20 次评估强制一次完整评估（防止 fast/full 统计漂移）
        #   3. GP 认为 EI 超过 top-20% 阈值 → 完整评估
        #   4. 其余 → 快速 AIG 评估
        if not self.multifidelity:
            use_full = True
        elif self.eval_count % 20 == 0:
            use_full = True
        elif self._ei_hint > self._ei_threshold:
            use_full = True
        else:
            use_full = False

        if use_full:
            stats = self.abc.run_sequence_and_stats(
                self.input_file, action_strs,
                self.mapping, self.map_arg, self.map_tail)
            self._full_eval_count += 1
        else:
            stats = self.abc.run_fast_stats(self.input_file, action_strs)
            self._fast_eval_count += 1

        # ---- 模块C: 记录电路特征（无论评估是否成功，必须追加）----
        init = self.init_stats
        if stats.get("nodes", -1) > 0:
            feat = np.array([
                stats["nodes"]  / max(init["nodes"],  1),
                stats["levels"] / max(init["levels"], 1),
                stats.get("edges", init.get("edges", 1))
                    / max(init.get("edges", 1), 1),
            ], dtype=np.float64)
        else:
            # 评估失败时用默认特征（无变化）
            feat = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        self._circuit_features.append(feat)

        # ---- 评估失败检查 ----
        if stats["nodes"] <= 0 and stats["area"] <= 0:
            return 10.0

        cost = self._compute_cost(stats)

        # ---- 模块F: 更新 Pareto 前沿 ----
        if (self.mapping
                and stats.get("area",  -1) > 0
                and stats.get("delay", -1) > 0):
            init_a = max(init.get("area",  1e-10), 1e-10)
            init_d = max(init.get("delay", 1e-10), 1e-10)
            self.pareto.update(
                stats["area"]  / init_a,
                stats["delay"] / init_d,
                list(indices), stats)
        elif stats.get("nodes", -1) > 0 and stats.get("levels", -1) > 0:
            self.pareto.update(
                stats["nodes"]  / max(init["nodes"],  1),
                stats["levels"] / max(init["levels"], 1),
                list(indices), stats)

        self._last_stats = stats
        self.eval_count += 1
        if cost < self.best_cost:
            self.best_cost  = cost
            self.best_seq   = action_strs
            self.best_stats = stats

        return cost


# ============================================================================
#  BOiLS 优化器主循环
#
#  Algorithm 1 (论文):
#    1. 随机采样 n0 个序列构建初始数据集 D0
#    2. for t = 0 … N_max:
#         a. 用 D_t 拟合 GP (SSK 核)
#         b. 在 TR(seq_best, ρ) 内最大化 EI → seq_{t+1}
#         c. 评估 QoR(seq_{t+1}), 更新 D, 更新 ρ
# ============================================================================

class BOiLSOptimizer:

    def __init__(self, evaluator: SynthesisEvaluator, seq_len: int,
                 n_actions: int, *, n_init=20, n_candidates=100,
                 ssk_order=2, seed=42, hp_interval=20,
                 enable_ple=False, init_seq_len=0,
                 batch_k=2, elite_size=15,
                 enable_cc_ssk=True, circuit_weight=0.3,
                 enable_seeded_init=True,
                 surrogate=None, surrogate_skip_delta=0.08,
                 ts_prob=0.5, diversity_thresh=0.8,
                 kernel_noise_var=1e-2):
        self.evaluator = evaluator
        self.n_actions = n_actions
        self.n_init = max(n_init, 5)
        self.n_cand = n_candidates
        self.hp_interval = hp_interval

        self.nop_idx = None
        for i, a in enumerate(evaluator.actions):
            if a == NOP:
                self.nop_idx = i
                break

        self.rng = np.random.default_rng(seed)

        # ---- 模块C: CC-SSK 核 ----
        _cw = circuit_weight if enable_cc_ssk else 0.0
        self.kernel = SSKKernel(subseq_order=ssk_order,
                                nop_idx=self.nop_idx,
                                circuit_weight=_cw,
                                noise_var=kernel_noise_var)
        self.gp = GaussianProcessSSK(self.kernel)

        # ---- 模块B: PLE 渐进长度扩展（默认关闭）----
        self.enable_ple = enable_ple
        self.seq_len = seq_len
        self.max_eff_len = seq_len          # 始终从全长开始
        if enable_ple:
            _init_len = (init_seq_len if init_seq_len > 0
                         else max(6, seq_len // 2))
            self.ple = ProgressiveLengthExpansion(
                init_len=_init_len, max_len=seq_len, step=3, patience=2)
            self.max_eff_len = self.ple.current_len
        else:
            self.ple = None

        self.min_eff_len = max(1, seq_len // 3)   # 下界固定为 seq_len//3

        # TrustRegion 始终以 seq_len 初始化
        self.tr = TrustRegion(seq_len, n_actions)

        # ---- 模块D: EGBO 精英池 ----
        self.batch_k = max(1, batch_k)
        self.elite_size = elite_size
        self.elite_pool = []   # list of (cost, seq)

        # ---- 模块A: 宏序列热启动开关 ----
        self.enable_seeded_init = enable_seeded_init

        self.X = []
        self._seq_feats = []   # 与 self.X 一一对应
        self.y_neg_cost = []
        self.best_x = None
        self.best_y = -float("inf")

        # 代理模型（可选）
        self.surrogate = surrogate
        self.surrogate_skip_delta = surrogate_skip_delta

        self.ts_prob = ts_prob
        self.diversity_thresh = diversity_thresh

        self._evaluated_set: set = set()   # 已评估序列的规范化 tuple 集合

    def _compute_seq_feat(self, seq: list) -> np.ndarray:
        """从序列索引提取结构特征，无需执行ABC，可用于任意候选序列。
        特征(9维): [frac_rewrite, frac_resub, frac_balance, frac_macro,
                   pw_rewrite, pw_resub, pw_balance, pw_macro, len_norm]
        前4维: 均匀统计；后4维: 线性位置加权（前段权重高）
        """
        acts = self.evaluator.actions
        n = len(seq)
        if n == 0:
            return np.zeros(9, dtype=np.float64)

        f_rewrite = sum(1 for i in seq if 'rewrite' in acts[i] or 'refactor' in acts[i]) / n
        f_resub = sum(1 for i in seq if 'resub' in acts[i]) / n
        f_balance = sum(1 for i in seq if 'balance' in acts[i] or 'dc2' in acts[i]
                         or 'dch' in acts[i]) / n
        f_macro = sum(1 for i in seq if acts[i] in MACRO_ACTIONS) / n

        weights = [(n - p) / n for p in range(n)]
        w_sum = sum(weights)
        pw_rewrite = sum(weights[p] for p, i in enumerate(seq)
                         if 'rewrite' in acts[i] or 'refactor' in acts[i]) / w_sum
        pw_resub = sum(weights[p] for p, i in enumerate(seq)
                       if 'resub' in acts[i]) / w_sum
        pw_balance = sum(weights[p] for p, i in enumerate(seq)
                         if 'balance' in acts[i] or 'dc2' in acts[i]
                         or 'dch' in acts[i]) / w_sum
        pw_macro = sum(weights[p] for p, i in enumerate(seq)
                      if acts[i] in MACRO_ACTIONS) / w_sum

        f_len = n / max(self.max_eff_len, 1)

        return np.array([f_rewrite, f_resub, f_balance, f_macro,
                         pw_rewrite, pw_resub, pw_balance, pw_macro,
                         f_len], dtype=np.float64)

    def _canonicalize(self, seq):
        if self.nop_idx is not None:
            return canonicalize_seq(seq, self.nop_idx)
        return list(seq)

    def _is_valid_seq(self, seq: list) -> bool:
        """dch 及其 flag 指令后不能跟 rewrite/resub/refactor/&dsdb（去 NOP 后检查）。"""
        acts = self.evaluator.actions
        eff = [int(a) for a in seq
               if self.nop_idx is None or int(a) != self.nop_idx]
        for i in range(1, len(eff)):
            p = acts[eff[i - 1]]
            c = acts[eff[i]]
            if "dch" in p and (
                "rewrite" in c or "resub" in c
                or "refactor" in c or "&dsdb" in c
            ):
                return False
        return True

    def _surr_to_cost(self, surr_pred) -> float:
        """将代理模型预测的 [area, delay] 转换为与 _compute_cost 一致的 cost 标量。"""
        init = self.evaluator.init_stats
        opt = self.evaluator.optimize
        if self.evaluator.mapping and init.get("area", -1) > 0:
            a_r = float(surr_pred[0]) / max(init["area"], 1e-10)
            d_r = float(surr_pred[1]) / max(init["delay"], 1e-10)
        else:
            a_r = float(surr_pred[0]) / max(init.get("nodes", 1), 1)
            d_r = float(surr_pred[1]) / max(init.get("levels", 1), 1)
        if opt == "area":
            return a_r
        elif opt == "delay":
            return d_r
        else:
            return a_r * d_r

    def _rand_seq(self):
        for _ in range(200):
            eff_len = int(self.rng.integers(self.min_eff_len, self.max_eff_len + 1))
            if eff_len <= 0:
                return []
            seq = list(self.rng.integers(0, self.n_actions, size=eff_len))
            if self._is_valid_seq(seq):
                return seq
        return seq  # 200次均失败时直接返回（极低概率）

    def _normalize(self):
        ya = np.array(self.y_neg_cost)
        ym = ya.mean()
        ys = max(ya.std(), 1e-8)
        return ya, ym, ys

    def _gp_training_pack(self, yn):
        """GP 拟合用 (X, y, circuit_feats)；len(X)>150 时取最优50+最近50 子集。"""
        yn = np.asarray(yn, dtype=float)
        _keep = None
        _gp_X = self.X
        _gp_yn = yn
        if len(self.X) > 150:
            _best_idx = list(np.argsort(np.array(self.y_neg_cost, dtype=float))[-50:])
            _recent_idx = list(range(len(self.X) - 50, len(self.X)))
            _keep = sorted(set(_best_idx + _recent_idx))
            _gp_X = [self.X[i] for i in _keep]
            _gp_yn = yn[_keep]
        if self.kernel.circuit_weight > 0 and len(self._seq_feats) == len(self.X):
            if _keep is not None:
                cfeats = np.array([self._seq_feats[i] for i in _keep], dtype=np.float64)
            else:
                cfeats = np.array(self._seq_feats, dtype=np.float64)
        else:
            cfeats = None
        return _gp_X, _gp_yn, cfeats

    # ----------------------------------------------------------------
    #  模块A: 宏序列热启动
    # ----------------------------------------------------------------
    def _seeded_init_sequences(self, n: int) -> list:
        """
        生成带宏序列种子的初始采样集（替代纯随机初始化）。

        用 `a in MACRO_ACTIONS` 精确识别宏动作，避免误判含分号的原子动作
        （如 "&get -n; &dsdb; &put"）。

        各占 1/3 策略：
          1/3 - 宏重复：宏动作索引重复填满 seq_len 槽
          1/3 - 宏前缀：宏动作 + 随机后缀
          1/3 - 纯随机：保持探索性
        """
        macro_indices = [
            i for i, a in enumerate(self.evaluator.actions)
            if a in MACRO_ACTIONS   # 与全局 MACRO_ACTIONS 精确匹配
        ]

        seeds = []
        n_repeat = n // 3
        n_hybrid = n // 3

        if macro_indices:
            # 类型1：宏重复（循环复用宏索引列表）
            for k in range(n_repeat):
                mi = macro_indices[k % len(macro_indices)]
                eff_len = int(self.rng.integers(self.min_eff_len, self.max_eff_len + 1))
                seq = [mi] * eff_len
                seeds.append(seq)

            # 类型2：宏前缀 + 随机后缀
            for k in range(n_hybrid):
                mi = macro_indices[k % len(macro_indices)]
                eff_len = int(self.rng.integers(self.min_eff_len, self.max_eff_len + 1))
                seq = [mi] + list(
                    self.rng.integers(0, self.n_actions,
                                      size=max(0, eff_len - 1)))
                seeds.append(seq)

        # 类型3 / 兜底：纯随机补足
        while len(seeds) < n:
            seeds.append(self._rand_seq())

        # 统一过滤非法序列（dch 约束）
        seeds = [s if self._is_valid_seq(s) else self._rand_seq() for s in seeds]

        return seeds[:n]

    # ----------------------------------------------------------------
    #  模块D: 精英池管理 + 进化候选生成
    # ----------------------------------------------------------------
    def _update_elite(self, seq: list, cost: float):
        """更新精英池：SSK 相似度约束 + cost 截断到 elite_size。"""
        if self.elite_pool:
            sims = [self.kernel.normalized(seq, e[1]) for e in self.elite_pool]
            max_sim = max(sims)
            most_similar_idx = int(np.argmax(sims))
            if max_sim >= 0.85:
                if cost < self.elite_pool[most_similar_idx][0]:
                    self.elite_pool[most_similar_idx] = (cost, list(seq))
                    self.elite_pool.sort(key=lambda x: x[0])
                return
        self.elite_pool.append((cost, list(seq)))
        self.elite_pool.sort(key=lambda x: x[0])
        self.elite_pool = self.elite_pool[:self.elite_size]

    def _evo_candidates(self, n: int) -> list:
        """
        从精英池通过单点交叉 + 随机变异生成 n 个候选序列。
        精英池不足2个时退回随机生成。
        子代长度限制在 [min_eff_len, max_eff_len]。
        """
        if len(self.elite_pool) < 1:
            return [self._rand_seq() for _ in range(n)]

        cands = []
        n_elite = min(len(self.elite_pool), self.elite_size)

        for _ in range(n):
            if len(self.elite_pool) >= 2 and self.rng.random() < 0.6:
                # 单点交叉
                idx = self.rng.choice(n_elite, size=2, replace=False)
                p1 = self.elite_pool[int(idx[0])][1]
                p2 = self.elite_pool[int(idx[1])][1]
                min_len = min(len(p1), len(p2))
                if min_len >= 2:
                    pt = int(self.rng.integers(1, min_len))
                    child = list(p1[:pt]) + list(p2[pt:])
                else:
                    child = list(p1[:min_len])
            else:
                # 精英变异
                idx = int(self.rng.integers(0, n_elite))
                child = list(self.elite_pool[idx][1])

            # 随机目标长度，覆盖 [min_eff_len, max_eff_len]（不锁死在亲本长度）
            target_len = int(self.rng.integers(self.min_eff_len, self.max_eff_len + 1))
            child = child[:target_len]
            while len(child) < target_len:
                child.append(int(self.rng.integers(0, self.n_actions)))

            # 随机变异 1~3 个位置
            n_mut = int(self.rng.integers(1, 4))
            for _ in range(n_mut):
                pos = int(self.rng.integers(0, len(child)))
                child[pos] = int(self.rng.integers(0, self.n_actions))

            if not self._is_valid_seq(child):
                child = self._rand_seq()
            cands.append(child)

        return cands

    def _restart_center(self) -> list:
        """
        重启时选取新的搜索起点：
        - 80% 概率：从精英池中选与 best_x SSK 距离最大的序列（跳离当前区域）
        - 20% 概率：纯随机，保持探索性
        """
        if self.best_x is None:
            return self._rand_seq()
        if self.elite_pool and len(self.elite_pool) >= 2 and self.rng.random() < 0.8:
            sims = [self.kernel.normalized(self.best_x, e[1]) for e in self.elite_pool]
            most_distant_idx = int(np.argmin(sims))
            base = self.elite_pool[most_distant_idx][1]
            print(f"  [TR] 精英池引导重启，跳至距离={1 - sims[most_distant_idx]:.2f} 的精英序列")
            return list(base)
        return self._rand_seq()

    def _try_stagnation_ple_expand(self) -> bool:
        """
        停滞触发的 PLE 扩展：扩展 max_eff_len 并重启 TR（与 TR dead 内 on_tr_restart 独立）。
        返回是否实际发生了扩展。
        """
        if not (self.enable_ple and self.ple is not None and not self.ple.at_max):
            return False
        old_len = self.ple.current_len
        self.ple.current_len = min(
            self.ple.current_len + self.ple.step, self.ple.max_len)
        self.ple.n_expansions += 1
        self.ple.expansion_log.append(
            (self.ple.n_expansions, old_len, self.ple.current_len, -self.best_y))
        self.max_eff_len = self.ple.current_len
        self.min_eff_len = min(self.max_eff_len, self.min_eff_len)
        self.tr.seq_len = self.max_eff_len
        self.tr.restart()
        print(f"  [PLE] 停滞扩展 #{self.ple.n_expansions}: "
              f"max_eff_len {old_len} → {self.ple.current_len}")
        return True

    def run(self, n_iters, timeout=0):
        t0 = time.time()

        # ================================================================
        # Phase 1: 初始化采样
        # ================================================================
        label = "宏序列热启动" if self.enable_seeded_init else "随机初始化"
        print(f"[CircuitSyn] {label} ({self.n_init} 样本, "
              f"初始有效长度上界={self.max_eff_len}) ...")

        _init_pool_mult = 5   # 候选池扩大倍数
        _init_pool_size = self.n_init * _init_pool_mult

        if self.enable_seeded_init:
            _init_pool = self._seeded_init_sequences(_init_pool_size)
        else:
            _init_pool = [self._rand_seq() for _ in range(_init_pool_size)]

        if self.surrogate and self.surrogate.enabled and len(_init_pool) > self.n_init:
            _init_preds = self.surrogate.predict_batch(_init_pool)   # (N, 2)
            _init_prods = _init_preds[:, 0] * _init_preds[:, 1]
            _top_idx = np.argsort(_init_prods)[:self.n_init]
            init_seqs = [_init_pool[i] for i in _top_idx]
            print(f"[Surrogate] Phase1 代理引导：从 {_init_pool_size} 个候选中选出 top-{self.n_init}")
        else:
            init_seqs = _init_pool[:self.n_init]

        abc_phase1 = 0
        for seq in init_seqs:
            if timeout > 0 and time.time() - t0 > timeout:
                break
            _key = tuple(self._canonicalize(seq))
            if _key in self._evaluated_set:
                continue
            self._evaluated_set.add(_key)
            cost = self.evaluator(seq)
            abc_phase1 += 1
            nc = -cost
            # 相似度去重（不影响精英池和 best_y，只跳过 GP 数据加入）
            _add_to_gp = True
            if len(self.X) > 0:
                _max_sim = max(self.kernel.normalized(seq, x) for x in self.X[-30:])
                if _max_sim > 0.95:
                    _add_to_gp = False
            if _add_to_gp:
                self.X.append(seq)
                self._seq_feats.append(self._compute_seq_feat(seq))
                self.y_neg_cost.append(nc)
            self._update_elite(seq, cost)

            improved_init = nc > self.best_y
            if improved_init:
                self.best_y = nc
                self.best_x = list(seq)

            # 每轮打印
            star = "★" if improved_init else " "
            ev = self.evaluator
            st = getattr(ev, "_last_stats", {}) or {}
            area = st.get("area", st.get("nodes", -1))
            delay = st.get("delay", st.get("levels", -1))
            seq_str = "; ".join(ev.indices_to_strs(seq))
            print(
                f"  [{abc_phase1:02d}][init]{star} cost={cost:.6f}  "
                f"len={len(seq)}/{self.max_eff_len}  "
                f"area={area}  delay={delay}\n"
                f"         序列: {seq_str}"
            )

        print(f"[CircuitSyn] 初始化完成, best_cost={-self.best_y:.6f}")

        # ---- GP 伪观测热启动（仅当代理可用时）----
        if self.surrogate and self.surrogate.enabled:
            _n_pseudo = 300
            _pseudo_seqs = [self._rand_seq() for _ in range(_n_pseudo)]
            _pseudo_preds = self.surrogate.predict_batch(_pseudo_seqs)  # (N, 2)
            _pseudo_costs = np.array([self._surr_to_cost(p) for p in _pseudo_preds])
            _pseudo_nc = -_pseudo_costs                               # neg_cost
            ya_real, ym, ys = self._normalize()
            yn_real = (ya_real - ym) / ys
            _pseudo_yn = (_pseudo_nc - ym) / ys
            _yn_aug = np.concatenate([yn_real, _pseudo_yn])
            _orig_noise = self.kernel.signal_var
            _cc_save = self.kernel._circuit_feats
            self.kernel._circuit_feats = None
            self.kernel.set_params(self.kernel.theta_m, self.kernel.theta_g,
                                   signal_var=_orig_noise * 3.0)
            _X_aug = list(self.X) + _pseudo_seqs
            self.gp.fit(_X_aug, _yn_aug)
            self.kernel.set_params(self.kernel.theta_m, self.kernel.theta_g,
                                   signal_var=_orig_noise)
            self.kernel._circuit_feats = _cc_save

            print(f"[Surrogate] GP 伪观测热启动：注入 {_n_pseudo} 个虚拟数据点")

        if self.surrogate and self.surrogate.enabled:
            print(f"[Surrogate] 代理模型已启用，剪枝比={self.surrogate.safe_cut_ratio*100:.0f}%，"
                  f"skip_delta={self.surrogate_skip_delta}")
        remaining = n_iters - abc_phase1
        print(f"[CircuitSyn] GP(CC-SSK) + TR + EGBO 搜索 "
              f"(剩余预算 {remaining} 次评估) ...\n")

        # ================================================================
        # Phase 2: GP 引导的贝叶斯优化主循环
        # t 计数实际 ABC 评估次数（含 Phase 1；可与 len(X) 不同，因 GP 相似度去重）
        # ================================================================
        t = abc_phase1
        no_improve_rounds = 0
        stagnation_patience = 10  # 连续无改进轮数触发停滞扩展（短序列空间不宜过大）

        while t < n_iters:
            if timeout > 0 and time.time() - t0 > timeout:
                print("[CircuitSyn] 已达超时限制，提前终止")
                break

            # ---- 归一化 y ----
            ya, ym, ys = self._normalize()
            yn = (ya - ym) / ys

            # ---- 拟合 GP（大数据集子采样 + CC-SSK 特征对齐）----
            _gp_X, _gp_yn, cfeats = self._gp_training_pack(yn)
            self.kernel._circuit_feats = cfeats
            self.gp.fit(_gp_X, _gp_yn)

            # ---- 周期性超参数优化 ----
            # bo_step 按评估次数计（t 从 n_init 起计 BO 轮次）
            bo_step = t - self.n_init
            _min_data_for_hp = self.n_init + 10   # 至少积累 n_init+10 个真实点再做 HP 优化
            if (self.hp_interval > 0
                    and bo_step > 0
                    and len(self.X) >= _min_data_for_hp):
                _adaptive_interval = (self.hp_interval
                                      if self.tr.radius > self.max_eff_len // 2
                                      else max(self.hp_interval // 2, 5))
                if bo_step % _adaptive_interval == 0:
                    self.gp.optimize_hp(n_restarts=2, max_iter=5)
                    print(f"  [HP] θ_m={self.kernel.theta_m:.3f}  "
                          f"θ_g={self.kernel.theta_g:.3f}  "
                          f"σ²={self.kernel.signal_var:.3f}  "
                          f"TR_ρ={self.tr.radius}  max_eff_len={self.max_eff_len}")
                    ya, ym, ys = self._normalize()
                    yn = (ya - ym) / ys
                    _gp_X, _gp_yn, cfeats = self._gp_training_pack(yn)
                    self.kernel._circuit_feats = cfeats
                    self.gp.fit(_gp_X, _gp_yn)

            # ---- TR 收敛处理（含 PLE 扩展）----
            if self.tr.dead:
                if self.enable_ple and self.ple is not None:
                    new_center, expanded = self.ple.on_tr_restart(
                        self.best_x, -self.best_y,
                        self.n_actions, self.rng)
                    if expanded:
                        self.max_eff_len = self.ple.current_len
                        self.tr.seq_len = self.max_eff_len
                        # 先更新有效长度上界，再 restart，使 radius = 新上界
                        self.tr.restart()
                        self.tr.radius = self.max_eff_len
                        seq = new_center   # new_center 不为 None（at_max时不扩展）
                    else:
                        self.tr.restart()
                        if self.rng.random() < 0.3:
                            seq = self._rand_seq()
                            self.best_x = list(seq)
                        else:
                            seq = self._rand_seq()
                else:
                    print(f"  [TR] 半径 → 0，重启 (第 {self.tr.restarts+1} 次)")
                    self.tr.restart()
                    seq = self._restart_center()
                    self.best_x = list(seq)

                # 注入 EI hint（重启时无 GP 预测，强制完整评估）
                self.evaluator._ei_hint = 1.0
                self.evaluator._ei_threshold = 0.5
                _key = tuple(self._canonicalize(seq))
                if _key in self._evaluated_set:
                    seq = self._rand_seq()
                    _key = tuple(self._canonicalize(seq))
                self._evaluated_set.add(_key)
                cost = self.evaluator(seq)
                nc = -cost
                self.X.append(seq)
                self._seq_feats.append(self._compute_seq_feat(seq))
                self.y_neg_cost.append(nc)
                self._update_elite(seq, cost)
                improved_tr = nc > self.best_y
                if improved_tr:
                    self.best_y = nc
                    self.best_x = list(seq)
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1
                t += 1
                if (no_improve_rounds >= stagnation_patience
                        and self._try_stagnation_ple_expand()):
                    no_improve_rounds = 0
                continue

            # ---- EGBO: 生成候选（TR随机 + 进化候选）----
            if self.best_x is None:
                self.best_x = self._rand_seq()
            tr_cands = [
                c if self._is_valid_seq(c) else self._rand_seq()
                for c in self.tr.sample(self.best_x, self.n_cand, self.rng)
            ]
            evo_cands = self._evo_candidates(self.n_cand * 2)
            rand_cands = [self._rand_seq() for _ in range(self.n_cand)]
            all_cands = tr_cands + evo_cands + rand_cands

            # ---- 代理模型剪枝：去掉预测最差的 30% ----
            if self.surrogate and self.surrogate.enabled and len(all_cands) > self.batch_k * 4:
                _surr_preds = self.surrogate.predict_batch(all_cands)   # (N, 2)
                _surr_prods = _surr_preds[:, 0] * _surr_preds[:, 1]
                _n_keep = max(self.batch_k * 4, int(len(all_cands) * 0.70))
                _keep_idx = np.argsort(_surr_prods)[:_n_keep]
                # 同步记录 rand 段在剪枝后的位置，用于 rand_best 选取
                _orig_rand_start = len(tr_cands) + len(evo_cands)
                _rand_positions = [new_i for new_i, orig_i in enumerate(_keep_idx)
                                   if orig_i >= _orig_rand_start]
                all_cands = [all_cands[i] for i in _keep_idx]
            else:
                _rand_positions = list(range(len(tr_cands) + len(evo_cands), len(all_cands)))

            # ---- GP 打分 ----
            all_feats = [self._compute_seq_feat(c) for c in all_cands]
            mu, var = self.gp.predict(all_cands, X_new_feats=all_feats)
            yb_n = (self.best_y - ym) / ys
            ei = expected_improvement(mu, var, yb_n)

            # 注入 EI hint（保持多保真度功能不变）
            ei_max = float(np.max(ei))
            ei_p80 = float(np.percentile(ei, 80))
            self.evaluator._ei_hint = ei_max
            self.evaluator._ei_threshold = ei_p80

            # ---- Thompson Sampling or EI 选候选 ----
            top_k = min(self.batch_k, len(all_cands))

            if self.rng.random() < self.ts_prob:
                ts_sample = mu + np.sqrt(np.maximum(var, 0.0)) * self.rng.standard_normal(len(mu))
                acq = ts_sample
            else:
                acq = ei

            if _rand_positions:
                rand_best = _rand_positions[int(np.argmax(acq[_rand_positions]))]
            else:
                rand_best = int(np.argmax(acq))
            all_sorted = list(np.argsort(acq)[::-1])
            top_indices = [rand_best]
            for i in all_sorted:
                if len(top_indices) >= top_k:
                    break
                if i == rand_best:
                    continue
                candidate = all_cands[i]
                too_similar = any(
                    self.kernel.normalized(candidate, all_cands[j]) > self.diversity_thresh
                    for j in top_indices
                )
                if not too_similar:
                    top_indices.append(int(i))

            if len(top_indices) < top_k:
                for i in all_sorted:
                    if len(top_indices) >= top_k:
                        break
                    if i not in top_indices:
                        top_indices.append(int(i))

            for idx in top_indices:
                if t >= n_iters:
                    break
                pick = all_cands[int(idx)]
                pick_list = list(pick)

                _key = tuple(self._canonicalize(pick_list))
                if _key in self._evaluated_set:
                    t += 1
                    continue
                self._evaluated_set.add(_key)

                # ---- 代理+GP 联合判断：是否跳过 ABC ----
                _surr_skip = False
                _skip_cost = None
                if (self.surrogate and self.surrogate.enabled
                        and self.best_y > -float("inf")):
                    best_cost_now = -self.best_y
                    # GP 预测 cost（反归一化 mu → 原始 neg_cost → cost）
                    _gp_neg = float(mu[int(idx)]) * ys + ym    # 归一化 neg_cost → 实际 neg_cost
                    _gp_cost = max(-_gp_neg, 1e-10)
                    # 代理模型预测 cost
                    _sp = self.surrogate.predict_batch([pick_list])[0]  # [area, delay]
                    _sc = self._surr_to_cost(_sp)
                    # 加权平均：GP 和代理各占 50%
                    _combined = 0.5 * _gp_cost + 0.5 * _sc
                    if _combined > best_cost_now + self.surrogate_skip_delta:
                        _surr_skip = True
                        _skip_cost = _sc

                if _surr_skip:
                    # 跳过 ABC，且不写入 GP 训练集（代理预测不作监督信号）
                    cost = _skip_cost
                    nc = -cost
                    # 不调用 _update_elite / 不更新 best_y（代理结果不可信作为最优）
                    improved = False
                    no_improve_rounds += 1
                    self.tr.update(False)
                    t += 1
                    seq_str = "; ".join(self.evaluator.indices_to_strs(pick_list))
                    print(
                        f"  [{t:02d}][surr]  cost={_sc:.6f}  "
                        f"len={len(pick_list)}/{self.max_eff_len}  "
                        f"area={_sp[0]:.2f}  delay={_sp[1]:.2f}  "
                        f"(gp={_gp_cost:.4f} surr={_sc:.4f} combined={_combined:.4f} skip_thr={best_cost_now + self.surrogate_skip_delta:.4f})\n"
                        f"         序列: {seq_str}"
                    )
                    if (no_improve_rounds >= stagnation_patience
                            and self._try_stagnation_ple_expand()):
                        no_improve_rounds = 0
                    continue

                # ---- 正常 ABC 评估（原有代码）----
                cost = self.evaluator(pick)
                nc = -cost
                self._update_elite(pick, cost)

                improved = nc > self.best_y
                if improved:
                    self.best_y = nc
                    self.best_x = list(pick)
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1

                # 相似度去重（不影响精英池和 best_y，只跳过 GP 数据加入）
                _add_to_gp = True
                if len(self.X) > 0:
                    _max_sim = max(self.kernel.normalized(pick_list, x) for x in self.X[-30:])
                    if _max_sim > 0.95:
                        _add_to_gp = False
                if _add_to_gp:
                    self.X.append(pick_list)
                    self._seq_feats.append(self._compute_seq_feat(pick_list))
                    self.y_neg_cost.append(nc)

                self.tr.update(improved)
                t += 1
                # 每轮打印：标注是否改善
                star = "★" if improved else " "
                ev = self.evaluator
                st = ev.best_stats if improved else (
                    ev._last_stats if hasattr(ev, "_last_stats") and ev._last_stats else {}
                )
                area = st.get("area", st.get("nodes", -1))
                delay = st.get("delay", st.get("levels", -1))
                seq_str = "; ".join(ev.indices_to_strs(pick))
                print(
                    f"  [{t:02d}][abc]{star} cost={cost:.6f}  "
                    f"len={len(pick_list)}/{self.max_eff_len}  "
                    f"area={area}  delay={delay}\n"
                    f"         序列: {seq_str}"
                )

                if (no_improve_rounds >= stagnation_patience
                        and self._try_stagnation_ple_expand()):
                    no_improve_rounds = 0

        # ================================================================
        # 结束
        # ================================================================
        elapsed = time.time() - t0
        print(f"\n[CircuitSyn] 搜索结束, 耗时 {elapsed:.1f}s "
              f"({elapsed/60:.1f}min)")
        if self.enable_ple and self.ple and self.ple.n_expansions > 0:
            log = " → ".join(
                str(e[1]) for e in self.ple.expansion_log
            ) + f" → {self.max_eff_len}"
            print(f"[PLE] 共 {self.ple.n_expansions} 次长度扩展: {log}")

        return self.best_x, -self.best_y


# ============================================================================
#  结果保存与展示
# ============================================================================

def save_results(evaluator: SynthesisEvaluator, optimizer: BOiLSOptimizer,
                 output_file: str, n_trials: int):
    seq_str = "; ".join(evaluator.best_seq)

    verify_parts = evaluator.abc._lib_read_cmds() + [f'read {evaluator.input_file}']
    if evaluator.abc._needs_strash(evaluator.input_file):
        verify_parts.append('strash')
    verify_parts.append(seq_str)
    if evaluator.mapping == "FPGA":
        if evaluator.map_tail:
            verify_parts.append(evaluator.map_tail)
        else:
            k = evaluator.map_arg if evaluator.map_arg else "6"
            verify_parts.append(f'if -K {k}')
    elif evaluator.mapping == "SCL":
        if evaluator.map_tail:
            verify_parts.append(evaluator.map_tail)
        else:
            verify_parts.append(f'map -D {evaluator.map_arg}' if evaluator.map_arg else 'map')
            verify_parts.append('topo')
    verify_parts.append('print_stats')
    abc_verify_cmd = "; ".join(verify_parts)

    results = {
        "best_cost": evaluator.best_cost,
        "best_sequence": evaluator.best_seq,
        "best_sequence_str": seq_str,
        "abc_verify_cmd": abc_verify_cmd,
        "best_stats": evaluator.best_stats,
        "init_stats": evaluator.init_stats,
        "improvement": {
            "nodes": f"{(1 - evaluator.best_stats.get('nodes', 0) / max(evaluator.init_stats.get('nodes', 1), 1)) * 100:.2f}%",
            "levels": f"{(1 - evaluator.best_stats.get('levels', 0) / max(evaluator.init_stats.get('levels', 1), 1)) * 100:.2f}%",
            "area": f"{(1 - evaluator.best_stats.get('area', 0) / max(evaluator.init_stats.get('area', 1e-10), 1e-10)) * 100:.2f}%",
            "delay": f"{(1 - evaluator.best_stats.get('delay', 0) / max(evaluator.init_stats.get('delay', 1e-10), 1e-10)) * 100:.2f}%",
        },
        "n_trials": n_trials,
        "n_evaluated": evaluator.eval_count,
        "search_config": {
            "method": "BOiLS (GP + SSK + Trust Region)",
            "best_seq_len": len(evaluator.best_seq),
            "param_dim": evaluator.seq_len,
            "n_actions": len(evaluator.actions),
            "actions": [a for a in evaluator.actions if a != NOP],
            "optimize": evaluator.optimize,
            "ssk_order": getattr(optimizer.kernel, "order", None),
            "final_theta_m": optimizer.kernel.theta_m,
            "final_theta_g": optimizer.kernel.theta_g,
            "tr_restarts": optimizer.tr.restarts,
            "enable_ple": optimizer.enable_ple,
            "enable_cc_ssk": optimizer.kernel.circuit_weight > 0,
            "circuit_weight": float(optimizer.kernel.circuit_weight),
            "batch_k": optimizer.batch_k,
            "multifidelity": evaluator.multifidelity,
            "final_seq_len": optimizer.max_eff_len,
        },
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"搜索完成! 结果已保存到: {output_file}")
    print(f"{'='*60}")
    print(f"总试验次数:  {n_trials} (实际评估 {evaluator.eval_count})")
    print(f"最优 cost:   {evaluator.best_cost:.6f}")
    init = evaluator.init_stats
    best = evaluator.best_stats
    if evaluator.mapping == "SCL":
        print(f"初始统计:    area={init.get('area', 'N/A')}  delay={init.get('delay', 'N/A')}  "
              f"nodes={init['nodes']}  levels={init['levels']}")
        print(f"最优统计:    area={best.get('area', 'N/A')}  delay={best.get('delay', 'N/A')}  "
              f"nodes={best.get('nodes', 'N/A')}  levels={best.get('levels', 'N/A')}")
        print(f"area 改善:   {results['improvement']['area']}")
        print(f"delay 改善:  {results['improvement']['delay']}")
    else:
        print(f"初始统计:    nodes={init['nodes']}  levels={init['levels']}")
        print(f"最优统计:    nodes={best.get('nodes', 'N/A')}  levels={best.get('levels', 'N/A')}")
        print(f"nodes 改善:  {results['improvement']['nodes']}")
        print(f"levels 改善: {results['improvement']['levels']}")
    print(f"SSK 超参数:  θ_m={optimizer.kernel.theta_m:.3f}  θ_g={optimizer.kernel.theta_g:.3f}")
    print(f"TR 重启:     {optimizer.tr.restarts} 次")
    print(f"\n最优序列:")
    print(f"  {seq_str}")

    print(f"\n可直接在 ABC 中验证:")
    print(f"  {abc_verify_cmd}")

    # ---- 模块F: Pareto 前沿输出 ----
    if evaluator.pareto.points:
        print(evaluator.pareto.summary_str())
        pareto_file = output_file.replace(".json", "_pareto.json")
        with open(pareto_file, "w", encoding="utf-8") as f:
            json.dump(evaluator.pareto.to_dict_list(), f,
                      indent=2, ensure_ascii=False)
        print(f"Pareto 前沿已保存: {pareto_file}")
        results["pareto"] = evaluator.pareto.to_dict_list()

    # ---- 模块E: 多保真度统计 ----
    if evaluator.multifidelity:
        total = evaluator._fast_eval_count + evaluator._full_eval_count
        print(f"多保真度: 快速={evaluator._fast_eval_count}  "
              f"完整={evaluator._full_eval_count}  "
              f"快速占比={evaluator._fast_eval_count/max(total,1)*100:.1f}%")

    # ---- 模块B: PLE 扩展日志 ----
    if optimizer.enable_ple and optimizer.ple and optimizer.ple.n_expansions > 0:
        results["ple_log"] = optimizer.ple.expansion_log

    print(f"{'='*60}")


# ============================================================================
#  主程序
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="BOiLS: 基于 GP + SSK 核 + 信赖域的逻辑综合贝叶斯优化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--abc_exe", type=str, required=True,
                        help="ABC 可执行文件路径")
    parser.add_argument("--input_file", type=str, required=True,
                        help="输入电路文件路径，支持 AIG (.aig) 和 BLIF (.blif) 格式")
    parser.add_argument("--cell_lib", type=str, default="",
                        help="标准单元库文件路径，多个库用分号分隔，如 \"a.genlib;b.lib;c.super\"。"
                             "自动根据后缀选择 read_genlib / read_lib / read_super")

    parser.add_argument("--custom_actions", type=str, default="",
                        help="自定义动作列表，逗号分隔 (覆盖全部内置动作空间)")
    parser.add_argument("--no_macros", action="store_true",
                        help="禁用宏动作，仅使用 31 个原子操作")
    parser.add_argument("--seq_len", type=int, default=20,
                        help="固定参数维度（序列槽位数），NOP 自动跳过实现变长 (default: 20)")
    parser.add_argument("--optimize", type=str, default="mix",
                        choices=["area", "delay", "mix"],
                        help="优化目标: area/delay/mix (default: mix)")

    parser.add_argument("--mapping", type=str, default="",
                        choices=["", "FPGA", "SCL"],
                        help="技术映射方式: 空=仅AIG优化, FPGA, SCL")
    parser.add_argument("--map_arg", type=str, default="",
                        help="映射参数: FPGA 用 K 值 (default: 6), SCL 用 delay 约束")
    parser.add_argument("--map_tail", type=str, default="",
                        help="自定义映射尾部命令 (覆盖默认映射命令)")

    parser.add_argument("--n_trials", type=int, default=200,
                        help="贝叶斯优化总迭代次数 (default: 200)")
    parser.add_argument("--n_init", type=int, default=20,
                        help="随机初始化采样数 (default: 20)")
    parser.add_argument("--n_candidates", type=int, default=100,
                        help="每轮信赖域内的候选序列数 (default: 100)")
    parser.add_argument("--ssk_order", type=int, default=2,
                        help="SSK 子序列阶数 p, 越高捕获结构越精细但计算越慢 (default: 2)")
    parser.add_argument("--kernel_noise_var", type=float, default=1e-2,
                        help="SSK 核对角观测噪声方差 noise_var (default: 1e-2)")
    parser.add_argument("--hp_interval", type=int, default=20,
                        help="GP 超参优化基准间隔；前 n_init+10 个真实点之前不触发，"
                             "TR 半径较小时间隔减半（下限 5），0=禁用 (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (default: 42)")
    parser.add_argument("--output", type=str, default="",
                        help="结果输出 JSON 文件路径 (default: auto)")
    parser.add_argument("--timeout", type=int, default=0,
                        help="搜索总时间限制(秒)，0=不限制 (default: 0)")

    # ---- 模块A: 宏序列热启动 ----
    parser.add_argument("--no_seeded_init", action="store_true",
                        help="禁用宏序列热启动，退回纯随机初始化")

    # ---- 模块B: 渐进长度扩展 PLE（默认关闭，需 --ple 开启）----
    parser.add_argument("--ple", action="store_true",
                        help="启用渐进长度扩展（默认关闭，固定使用全长 seq_len）")
    parser.add_argument("--init_seq_len", type=int, default=0,
                        help="PLE 初始序列长度，0=自动(max(6,seq_len//2)) "
                             "(default: 0)")

    # ---- 模块C: CC-SSK 电路感知核 ----
    parser.add_argument("--no_cc_ssk", action="store_true",
                        help="禁用 CC-SSK，退回原版纯 SSK")
    parser.add_argument("--circuit_weight", type=float, default=0.3,
                        help="CC-SSK 电路相似度混合权重 [0,1] (default: 0.3)")

    # ---- 模块D: EGBO ----
    parser.add_argument("--batch_k", type=int, default=2,
                        help="每轮实际评估候选数 (1=原版单点BO, "
                             "2+=EGBO批量) (default: 2)")
    parser.add_argument("--elite_size", type=int, default=15,
                        help="精英池大小 (default: 15)")
    parser.add_argument("--ts_prob", type=float, default=0.5,
                        help="每轮使用 Thompson Sampling 的概率，其余用 EI (default: 0.5)")
    parser.add_argument("--diversity_thresh", type=float, default=0.8,
                        help="batch 内候选 SSK 相似度上限，越小越多样 (default: 0.8)")

    # ---- 模块E: 多保真度评估（默认关闭）----
    parser.add_argument("--multifidelity", action="store_true",
                        help="启用多保真度评估（低质量候选用快速AIG评估，"
                             "高质量候选用完整评估）。默认关闭。")
    parser.add_argument("--verbose", action="store_true",
                        help="显示详细搜索日志")

    # ---- 代理模型加速 ----
    parser.add_argument("--surrogate_ckpt_dir", type=str, default="",
                        help="代理模型 checkpoint 目录，空=禁用代理加速")
    parser.add_argument("--surrogate_aag", type=str, default="",
                        help="电路 AAG 文件路径（代理模型编码电路图用）")
    parser.add_argument("--surrogate_aag_dir", type=str, default="",
                        help="AAG 文件目录，自动拼接 <circuit_name>.aag（--surrogate_aag 优先）")
    parser.add_argument("--surrogate_csv", type=str, default="",
                        help="代理模型可靠性 CSV 路径（可选，空=跳过可靠性检查）")
    parser.add_argument("--surrogate_skip_delta", type=float, default=0.08,
                        help="联合预测比 best_cost 高出超过 delta 时跳过 ABC (default: 0.08)")

    return parser.parse_args()


def main():
    args = parse_args()

    abc_runner = ABCRunner(args.abc_exe, cell_libs=args.cell_lib)

    if args.custom_actions:
        actions = [a.strip() for a in args.custom_actions.split(",")]
        if NOP not in actions:
            actions.append(NOP)
    elif args.no_macros:
        actions = ATOMIC_ACTIONS + [NOP]
    else:
        actions = ACTIONS

    evaluator = SynthesisEvaluator(
        abc_runner=abc_runner,
        input_file=args.input_file,
        actions=actions,
        seq_len=args.seq_len,
        optimize=args.optimize,
        mapping=args.mapping,
        map_arg=args.map_arg,
        map_tail=args.map_tail,
        multifidelity=args.multifidelity,
    )

    surrogate = None
    if args.surrogate_ckpt_dir and _SURROGATE_AVAILABLE:
        input_path = Path(args.input_file)
        stem = input_path.stem
        if stem.endswith(".parmys"):
            base_stem = stem[: -len(".parmys")]
            # ckpt 优先尝试 base_stem（去掉 .parmys），再尝试完整 stem
            if Path(args.surrogate_ckpt_dir, f"iwls26_{base_stem}.pt").is_file():
                circuit_name = base_stem
            else:
                circuit_name = stem
        else:
            circuit_name = stem
        input_suffix = input_path.suffix.lower()  # e.g. ".blif" 或 ".aig"

        # ---- 解析 aag_path ----
        if args.surrogate_aag:
            # 显式指定，直接用
            aag_path = args.surrogate_aag
        elif args.surrogate_aag_dir:
            # 从目录自动推断：根据输入文件名决定候选顺序
            # 若文件名形如 *.parmys.blif，stem = "ac97_ctrl.parmys"
            # base_stem = 去掉 .parmys 后缀的部分，e.g. "ac97_ctrl"
            if stem.endswith(".parmys"):
                base_stem = stem[: -len(".parmys")]
                candidates = [
                    base_stem + ".aag",       # ac97_ctrl.aag
                    stem + ".aag",            # ac97_ctrl.parmys.aag
                    base_stem + ".aig",
                    stem + ".aig",
                ]
            else:
                candidates = [
                    stem + ".aag",
                    stem + ".aig",
                ]
            aag_path = ""
            for cand in candidates:
                full = str(Path(args.surrogate_aag_dir) / cand)
                if Path(full).is_file():
                    aag_path = full
                    print(f"[Surrogate] 自动找到 AAG 文件: {aag_path}")
                    break
            if not aag_path:
                print(f"[警告] 在 {args.surrogate_aag_dir} 未找到 {candidates}，代理加速已跳过")
        else:
            # 无任何 aag 参数：input_file 本身若是 aag/aig 则直接用，否则跳过
            if input_suffix in (".aag", ".aig"):
                aag_path = str(input_path)
            else:
                print(f"[警告] --surrogate_aag 未指定，input_file 为 {input_suffix} 非 AAG/AIG，"
                      f"请用 --surrogate_aag 或 --surrogate_aag_dir 指定电路图，代理加速已跳过")
                aag_path = ""

        if aag_path:
            surrogate = ModelSurrogate(
                circuit_name=circuit_name,
                aag_path=aag_path,
                ckpt_dir=args.surrogate_ckpt_dir,
                actions=actions,
                csv_path=args.surrogate_csv,
                device="cpu",
            )
    elif args.surrogate_ckpt_dir and not _SURROGATE_AVAILABLE:
        print("[警告] 指定了 --surrogate_ckpt_dir 但 model_surrogate 模块不可导入，代理加速已跳过")

    optimizer = BOiLSOptimizer(
        evaluator=evaluator,
        seq_len=args.seq_len,
        n_actions=len(actions),
        n_init=args.n_init,
        n_candidates=args.n_candidates,
        ssk_order=args.ssk_order,
        seed=args.seed,
        hp_interval=args.hp_interval,
        enable_ple=args.ple,
        init_seq_len=args.init_seq_len,
        batch_k=args.batch_k,
        elite_size=args.elite_size,
        enable_cc_ssk=not args.no_cc_ssk,
        circuit_weight=args.circuit_weight,
        enable_seeded_init=not args.no_seeded_init,
        surrogate=surrogate,
        surrogate_skip_delta=args.surrogate_skip_delta,
        ts_prob=args.ts_prob,
        diversity_thresh=args.diversity_thresh,
        kernel_noise_var=args.kernel_noise_var,
    )

    print(f"\n{'='*60}")
    print(f"CircuitSyn 搜索配置:")
    print(f"  迭代次数={args.n_trials}  SSK阶数={args.ssk_order}  "
          f"候选数={args.n_candidates}  HP间隔={args.hp_interval}")
    print(f"  PLE={'开启 (初始='+str(optimizer.max_eff_len)+')' if optimizer.enable_ple else '关闭'}  "
          f"CC-SSK={'开启 (w='+str(args.circuit_weight)+')' if not args.no_cc_ssk else '关闭'}  "
          f"EGBO={'开启 (K='+str(args.batch_k)+')' if args.batch_k > 1 else '关闭(K=1)'}  "
          f"多保真度={'开启' if args.multifidelity else '关闭'}")
    print(f"  加速后端={_SSK_BACKEND}"
          + (" (pip install numba 可获得 ~50-100x 加速)"
             if _SSK_BACKEND == "python" else ""))
    print(f"{'='*60}\n")

    optimizer.run(args.n_trials, timeout=args.timeout if args.timeout > 0 else 0)

    output_file = args.output
    if not output_file:
        stem = Path(args.input_file).stem
        output_file = f"boils_result_{stem}.json"

    save_results(evaluator, optimizer, output_file, args.n_trials)


if __name__ == "__main__":
    main()