from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Tuple, Set

import torch
import dgl
import aiger


# -----------------------------
# 1) 导入 py-aiger 的 Node 类（不同版本路径可能略有差异）
# -----------------------------
def _import_aiger_node_classes():
    """
    返回: (AIG, AndGate, Inverter, Input, LatchIn, ConstFalse)
    若 import 失败，只需在这里改一处路径即可。
    """
    try:
        from aiger.aig import AIG, AndGate, Inverter, Input, LatchIn, ConstFalse
        return AIG, AndGate, Inverter, Input, LatchIn, ConstFalse
    except Exception:
        try:
            from aiger import aig as aigmod
            return aigmod.AIG, aigmod.AndGate, aigmod.Inverter, aigmod.Input, aigmod.LatchIn, aigmod.ConstFalse
        except Exception as e:
            raise ImportError(
                "无法导入 py-aiger 的 Node 类。请检查你的 aiger 版本与模块路径，"
                "并修改 _import_aiger_node_classes() 中的 import 路径。"
            ) from e


AIG, AndGate, Inverter, Input, LatchIn, ConstFalse = _import_aiger_node_classes()


# -----------------------------
# 2) 5类节点类型 one-hot 编码（按你要求：latch/and/not/PI/PO）
# -----------------------------
NTYPE = {"PI": 0, "AND": 1, "NOT": 2, "LATCH": 3, "PO": 4}


def _node_type_idx(n: Any) -> int:
    if isinstance(n, AndGate):
        return NTYPE["AND"]
    if isinstance(n, Inverter):
        return NTYPE["NOT"]
    if isinstance(n, LatchIn):
        return NTYPE["LATCH"]
    if isinstance(n, Input):
        return NTYPE["PI"]
    if isinstance(n, ConstFalse):
        # 你没定义 CONST 类型：这里把 const0 当作 PI 类别（不会增加类型数）
        return NTYPE["PI"]
    if isinstance(n, str) and n.startswith("__PO__"):
        return NTYPE["PO"]
    return NTYPE["PI"]


# -----------------------------
# 3) 读入 .aag -> AIG 对象
# -----------------------------
def load_aig_from_aag(aag_path: str) -> AIG:
    circ = aiger.load(aag_path)
    return circ.aig if hasattr(circ, "aig") else circ


# -----------------------------
# 4) 建图：单一边类型，边 = comb_edges ∪ seq_edges；并加 PO 虚拟节点
# -----------------------------
def build_graph_from_aag(aag_path: str) -> Tuple[dgl.DGLGraph, Dict[str, Any]]:
    """
    返回:
      g: dgl.DGLGraph (同构图，单一边类型)
      meta: 含映射及用于 level 计算的 comb_edges（包含 driver->PO 边）
    """
    aig = load_aig_from_aag(aag_path)

    # roots = PO drivers + latch next-state roots (D cones)
    out_roots = list(aig.node_map.values())
    latch_roots = list(aig.latch_map.values())
    roots = out_roots + latch_roots

    # (A) 遍历组合 DAG：沿 children 收集所有 Node，并记录 comb 边（child -> parent）
    nodes: Set[Any] = set()
    comb_pairs: List[Tuple[Any, Any]] = []

    q = deque(roots)
    while q:
        n = q.popleft()
        if n in nodes:
            continue
        nodes.add(n)
        for ch in n.children:
            comb_pairs.append((ch, n))
            if ch not in nodes:
                q.append(ch)

    # (B) 确保每个 latch name 都有对应的 LatchIn 节点
    latch_name2node: Dict[str, LatchIn] = {}
    for n in list(nodes):
        if isinstance(n, LatchIn):
            latch_name2node[n.name] = n

    for lname in aig.latches:
        if lname not in latch_name2node:
            ln = LatchIn(lname)
            latch_name2node[lname] = ln
            nodes.add(ln)

    # (C) PO 虚拟节点 + 边：driver -> PO
    po_nodes: List[str] = []
    po_pairs: List[Tuple[Any, Any]] = []
    for oname, driver in aig.node_map.items():
        po = f"__PO__{oname}"
        po_nodes.append(po)
        po_pairs.append((driver, po))

    # (D) seq 边：next_state_root -> LatchIn(name)
    seq_pairs: List[Tuple[Any, Any]] = []
    for lname, next_root in aig.latch_map.items():
        seq_pairs.append((next_root, latch_name2node[lname]))

    # (E) 分配连续 node id（✅ 必须把 PO 节点加进 all_nodes）
    all_nodes: List[Any] = list(nodes) + po_nodes
    node2id: Dict[Any, int] = {n: i for i, n in enumerate(all_nodes)}

    # ✅ 总边必须包含 po_pairs
    all_pairs: List[Tuple[Any, Any]] = comb_pairs + po_pairs + seq_pairs

    src = torch.tensor([node2id[s] for s, _ in all_pairs], dtype=torch.int64)
    dst = torch.tensor([node2id[t] for _, t in all_pairs], dtype=torch.int64)

    g = dgl.graph((src, dst), num_nodes=len(all_nodes))

    meta = {
        "aig": aig,
        "all_nodes": all_nodes,
        "node2id": node2id,
        "id2node": {i: n for n, i in node2id.items()},
        # ✅ level 只用 DAG 边算：comb + driver->PO（不含 seq）
        "comb_pairs_for_level": comb_pairs + po_pairs,
        "all_pairs": all_pairs,
        "seq_pairs": seq_pairs,
        "po_pairs": po_pairs,
        "po_nodes": po_nodes,
    }
    return g, meta


# -----------------------------
# 5) 计算节点特征 nf：one-hot(5) + fanin + fanout + level
# -----------------------------
def compute_node_nf(g: dgl.DGLGraph, meta: Dict[str, Any]) -> torch.Tensor:
    """
    nf 维度 = 5(one-hot) + 1(fanin) + 1(fanout) + 1(level) = 8

    fanin/fanout：基于 g 的所有边（comb+seq 已合并）
    level：只在组合 DAG 边上计算（comb_pairs_for_level），保证 latch 后第一个门 level=1
    """
    all_nodes: List[Any] = meta["all_nodes"]
    node2id: Dict[Any, int] = meta["node2id"]
    N = len(all_nodes)

    # (A) one-hot
    t_idx = torch.tensor([_node_type_idx(n) for n in all_nodes], dtype=torch.int64)
    one_hot = torch.zeros((N, 5), dtype=torch.float32)
    one_hot[torch.arange(N), t_idx] = 1.0

    # (B) fanin/fanout：直接用图的入度/出度（包含 comb+seq）
    fanin = g.in_degrees().to(torch.float32).view(-1, 1)
    fanout = g.out_degrees().to(torch.float32).view(-1, 1)

    # (C) level：只用 DAG 边算（否则 seq 会形成环）
    dag_pairs: List[Tuple[Any, Any]] = meta["comb_pairs_for_level"]

    succ = [[] for _ in range(N)]
    indeg = [0] * N

    for s, t in dag_pairs:
        sid = node2id[s]
        tid = node2id[t]
        succ[sid].append(tid)
        indeg[tid] += 1

    level = torch.zeros(N, dtype=torch.float32)
    dq = deque([i for i in range(N) if indeg[i] == 0])

    # Kahn topo + DP
    while dq:
        u = dq.popleft()
        for v in succ[u]:
            level[v] = torch.maximum(level[v], level[u] + 1.0)
            indeg[v] -= 1
            if indeg[v] == 0:
                dq.append(v)

    level = level.view(-1, 1)

    # 拼接 nf
    nf = torch.cat([one_hot, fanin, fanout, level], dim=1)  # (N, 8)
    return nf


# -----------------------------
# 6) 一键调用：aag -> DGLGraph + g.ndata["nf"]
# -----------------------------
def aag_to_dgl_graph(aag_path: str) -> Tuple[dgl.DGLGraph, Dict[str, Any]]:
    """
    返回:
      g: 单一边类型 DGLGraph（边无特征）
      meta: 映射/调试信息（含 nf 张量）
    """
    g, meta = build_graph_from_aag(aag_path)
    nf = compute_node_nf(g, meta)
    g.ndata["nf"] = nf
    meta["nf"] = nf
    meta["nf_dim"] = nf.shape[1]
    return g, meta


# -----------------------------
# 7) 示例
# -----------------------------
if __name__ == "__main__":
    def gen_simple_seq_aag(aag_path: str = "toy_seq.aag"):
        # 1) 建一个组合电路：y = ~(a & q)
        a, q = aiger.atoms("a", "q")
        y_expr = ~(a & q)

        # 给输出命名为 'y'，得到一个 AIG 电路对象
        comb = y_expr.with_output("y").aig

        # 2) 用 loopback 把输出 y 延迟反馈到输入 q，生成 latch（名字也叫 'q'）
        # README: loopback 的字典键包括 input/output/latch/init/keep_output 等 :contentReference[oaicite:1]{index=1}
        seq = comb.loopback({
            "input": "q",      # 把输入端口 q 变成“延迟输入”
            "output": "y",     # 从输出 y 回馈
            "latch": "q",      # latch 名称
            "init": False,     # 初始值
            "keep_output": True
        })

        # 3) 写出 aag 文件
        seq.write(aag_path)
        return aag_path


    def read_and_check(aag_path: str):
        circ = aiger.load(aag_path)   # 读 .aag
        aig = circ.aig                # 取出 AIG

        print("=== Basic interface sets ===")
        print("inputs  :", sorted(aig.inputs))
        print("outputs :", sorted(aig.outputs))
        print("latches :", sorted(aig.latches))

        print("\n=== AIGER header preview ===")
        # repr(aig) 会 dump 成 AIGER 文本
        aiger_text = repr(aig).splitlines()
        for line in aiger_text[:8]:
            print(line)

        print("\n=== node_map / latch_map keys ===")
        print("node_map (PO names):", list(aig.node_map.keys()))
        print("latch_map (latch names):", list(aig.latch_map.keys()))

        # 4) 简单仿真 2 拍确认 latch 在工作
        # 第 0 拍：q = init(False)
        # 输入 a=1 => y = ~(1 & 0) = 1; 下一拍 q_next=1
        # 第 1 拍：q=1
        # 输入 a=1 => y = ~(1 & 1) = 0; 下一拍 q_next=0
        sim = aig.simulator()
        next(sim)
        print("\n=== simulate ===")
        print("t=0:", sim.send({"a": True}))   # 返回 (outs, latches_next)
        print("t=1:", sim.send({"a": True}))

    path = gen_simple_seq_aag("toy_seq.aag")
    read_and_check(path)
    g, meta = aag_to_dgl_graph(path)
    print("=== DGL Graph ===")
    print(g)
    print("num_nodes:", g.num_nodes())
    print("num_edges:", g.num_edges())

    # 打印 nf 的具体值
    print("\n=== g.ndata['nf'] ===")
    nf = g.ndata["nf"]
    print("nf shape:", tuple(nf.shape))
    # 每个节点一行
    for i in range(g.num_nodes()):
        print(f"node {i:3d} nf = {nf[i].tolist()}")

    # 打印边 (src, dst)
    print("\n=== g.edges() ===")
    src, dst = g.edges()
    src = src.tolist()
    dst = dst.tolist()
    print(f"total edges: {len(src)}")
    for e, (s, t) in enumerate(zip(src, dst)):
        print(f"edge {e:3d}: {s} -> {t}")

    # （可选）把 meta 里的 node 映射也打印出来，方便你对照（不需要可删）
    print("\n=== meta id2node (for debugging) ===")
    for i in range(g.num_nodes()):
        print(f"id {i:3d}: {meta['id2node'][i]}")