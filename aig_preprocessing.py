import numpy as np
import networkx as nx
import aigverse.adapters 
import dgl
import torch
import networkx as nx
from aigverse import read_aiger_into_aig
from typing import Optional, Union


def expand_inverted_edges_to_not_nodes(G: nx.DiGraph) -> nx.DiGraph:
    """
    扫描 G 中的所有边，如果特征为 [0, 1]，则是原始边，直接添加到 H 中；
    如果特征为 [1, 0]，则需要插入 NOT 节点，并且展开边 u -> not_node -> v；
    其他情况报错。对于多个目标相同的反向边，合并为一个 NOT 节点。
    同时修改节点 0 的 type 为 [0, 0, 0, 0]，NOT 节点的 type 为 [1, 0, 0, 0]。
    type特征：[0, 1, 0, 0] 表示输入节点，[1, 0, 0, 0] 表示 NOT 节点，[0, 0, 0, 1] 表示输出节点，[0, 0, 1, 0] 表示and节点。
    """
    H = nx.DiGraph()  # 新图
    not_id = 0
    not_node_map = {}  # 用来存储每个目标节点对应的 NOT 节点

    # 复制所有节点到新图，保持原有的 type 属性
    for n, attr in G.nodes(data=True):
        H.add_node(n, **attr)

    # 遍历 G 中的所有边
    for u, v, eattr in G.edges(data=True):
        # 获取边的特征
        feature = eattr.get("type")

        # 使用 np.array_equal 来比较 NumPy 数组
        if np.array_equal(feature, [1, 0]):
            # 普通边，直接保留
            H.add_edge(u, v)
        elif np.array_equal(feature, [0, 1]):
            # 反向边，插入 NOT 节点
            # 如果已经有一个反向边指向相同目标节点 v，使用相同的 NOT 节点
            if u not in not_node_map:
                # 插入新的 NOT 节点
                not_node = f"not_{not_id}"
                not_id += 1
                not_node_map[u] = not_node

                # 插入 NOT 节点，设置 type 为 [1, 0, 0, 0]
                H.add_node(not_node)
                H.nodes[not_node]["type"] = [1, 0, 0, 0]
                H.add_edge(u, not_node)  # 插入 u -> not_node
            
            # 获取对应目标节点 v 的 NOT 节点
            not_node = not_node_map[u]

            # 插入新的边：not_node -> v
            H.add_edge(not_node, v)  # 插入 not_node -> v
        else:
            # 其他情况，报错
            raise ValueError(f"Unexpected edge feature {feature} between nodes {u} and {v}")

    # 修改节点 0 的 type 为 [0, 0, 0, 0]
    if 0 in H.nodes:
        H.nodes[0]['type'] = [0, 0, 0, 0]

    return H


def compute_level_longest_path_from_pis(G: nx.DiGraph) -> dict:
    """
    计算 logic level：
      - PI level = 0（入度为 0 的节点视为 PI）
      - 其他节点 level = 1 + max(level(fanin))
    注意：G 必须是 DAG
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph is not a DAG; cannot compute levels with longest-path DP.")

    topo = list(nx.topological_sort(G))
    level = {}

    for n in topo:
        if G.in_degree(n) == 0:
            level[n] = 0
        else:
            level[n] = 1 + max(level[u] for (u, _) in G.in_edges(n))

    return level


def load_aig_as_dgl(
    aig_path: str,
    dtype=np.int32,
    device: Optional[Union[str, torch.device]] = None,
) -> dgl.DGLGraph:
    """
    输入：aig/aag 文件路径
    输出：DGLGraph
      - g.ndata['nf']: (N, 7) float32
      - 删除孤立节点（fanin=0 且 fanout=0）
      - 会把 nx 节点重编号为 0..N-1，保证特征对齐
    """
    aig = read_aiger_into_aig(aig_path)

    G0 = aig.to_networkx(
        levels=False,
        fanouts=False,
        node_tts=False,
        dtype=dtype
    )

    # 展开反向边为 NOT 节点
    H = expand_inverted_edges_to_not_nodes(G0)

    # 删除孤立节点
    isolated = [n for n in H.nodes() if H.in_degree(n) == 0 and H.out_degree(n) == 0]
    if isolated:
        H.remove_nodes_from(isolated)
        # 大数据训练时不建议频繁 print；如果你要日志，建议改成 logging 并设定频率
        # print(f"[INFO] removed isolated nodes: {len(isolated)}")

    # 计算 level / fanin / fanout
    levels = compute_level_longest_path_from_pis(H)
    fanin = {n: H.in_degree(n) for n in H.nodes}
    fanout = {n: H.out_degree(n) for n in H.nodes}

    # 生成每个节点的 nf（7维）
    # 注意：不要在这里 clear() 再写回，直接生成 nf 映射即可（更安全）
    nf_dict = {}
    for n in H.nodes:
        t = H.nodes[n].get("type", None)
        if t is None:
            raise ValueError(f"Node {n} missing 'type' attribute")
        t = np.asarray(t, dtype=dtype).reshape(-1)
        if t.shape[0] != 4:
            raise ValueError(f"Node {n} type dim != 4, got {t.shape}")

        nf = np.concatenate(
            [t, np.asarray([levels[n], fanin[n], fanout[n]], dtype=dtype)],
            axis=0
        )
        if nf.shape[0] != 7:
            raise RuntimeError(f"Node {n} nf dim != 7")

        nf_dict[n] = nf

    # --- 关键：重编号节点为 0..N-1，保证 DGL 的 ndata 顺序与 nf 对齐 ---
    # relabel_nodes 会返回新图和 old->new 映射
    H2 = nx.relabel_nodes(H, {old: i for i, old in enumerate(H.nodes())}, copy=True)

    # 按新编号顺序构造 (N,7)
    N = H2.number_of_nodes()
    nf = np.zeros((N, 7), dtype=dtype)
    # mapping 是 old->new，但我们构造时用 old 的 nf 填到 new 位置
    # 我们自己构建的 mapping 与 H.nodes() 的顺序一致，所以也可以直接按 enumerate(H.nodes()) 填
    for old, new in {old: i for i, old in enumerate(H.nodes())}.items():
        nf[new] = nf_dict[old]

    # 建 DGLGraph（只用结构）
    g = dgl.from_networkx(H2)  # 节点已经是 0..N-1

    # 设置节点特征：一般用 float32 喂给 GNN
    g.ndata["nf"] = torch.from_numpy(nf).to(torch.float32)
    if device is not None:
        g = g.to(device)

    return g


def dump_node_edges_in_G0(G0: nx.DiGraph, n, max_edges=50):
    print(f"\n===== Dump edges in G0 for node {n} =====")
    if n not in G0:
        print(f"[INFO] node {n} not in G0 (likely a inserted NOT node).")
        return

    print("G0 node attr:", G0.nodes[n])

    in_es = list(G0.in_edges(n, data=True))
    out_es = list(G0.out_edges(n, data=True))

    print(f"#in_edges={len(in_es)}, #out_edges={len(out_es)}")

    print("  in_edges (u -> n):")
    for i, (u, v, ea) in enumerate(in_es[:max_edges]):
        print(f"    {u} -> {v}, attr={ea}")
    if len(in_es) > max_edges:
        print(f"    ... ({len(in_es)-max_edges} more)")

    print("  out_edges (n -> v):")
    for i, (u, v, ea) in enumerate(out_es[:max_edges]):
        print(f"    {u} -> {v}, attr={ea}")
    if len(out_es) > max_edges:
        print(f"    ... ({len(out_es)-max_edges} more)")


if __name__ == "__main__":
    G = load_aig_as_dgl("/home/yfdai/asap/data/aig/yosys.aig", dtype=np.int32)

    nf = G.ndata["nf"]
    print("nf shape:", nf.shape)

    