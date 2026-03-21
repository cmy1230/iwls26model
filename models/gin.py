import torch
import torch.nn as nn
import dgl
import networkx as nx
import torch.nn.functional as F
from dgl.nn import GINConv

class GIN(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers):
        super(GIN, self).__init__()
        
        # 定义一个列表来存储所有的 GIN 层
        self.layers = nn.ModuleList()
        
        # 第 1 层：输入维度 input_dim
        # 每层都使用一个 MLP 来处理特征
        for i in range(num_layers):
            if i == 0:
                # 第一层：输入维度是 input_dim
                mlp = nn.Sequential(
                    nn.Linear(input_dim, output_dim),  # 第一层线性映射
                    nn.ReLU(),
                    nn.Linear(output_dim, output_dim)         # 第二层线性映射
                )
            else:
                mlp = nn.Sequential(
                    nn.Linear(output_dim, output_dim),
                    nn.ReLU(),
                    nn.Linear(output_dim, output_dim)
                )
            
            # 使用 DGL 的 GINConv，设定 learn_eps=True 使得 epsilon 可训练
            self.layers.append(GINConv(mlp, aggregator_type='sum', init_eps=0, learn_eps=True))

    def forward(self, g, h):
        # 逐层传递节点特征
        for layer in self.layers:
            h = layer(g, h)
        return h


if __name__ == "__main__":

    G = nx.erdos_renyi_graph(10, 0.5)  # 生成一个10节点的随机图
    # 转换为DGL图
    g = dgl.from_networkx(G)

    # 定义输入、输出维度和层数
    input_dim = 7  # 输入特征的维度
    output_dim = 128  # 输出特征的维度
    num_layers = 3  # 网络层数

    # 初始化GIN模型
    gin = GIN(input_dim, output_dim, num_layers)

    # 为每个节点随机生成初始特征
    node_features = torch.randn(g.num_nodes(), input_dim)

    # 进行前向传播
    output = gin(g, node_features)

    print(output.shape)  # 输出的shape是 (num_nodes, output_dim)
