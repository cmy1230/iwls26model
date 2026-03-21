import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractionLayer(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(FeatureExtractionLayer, self).__init__()
        
        self.num_heads = num_heads
        
        # 多头自注意力（MHA）
        self.mha = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        
        # MLP 用于特征投影
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        
    def forward(self, x):
        # 输入 x 的形状为 (batch_size, input_dim)，每个样本的特征维度为 input_dim
        
        # 将输入扩展为适合 MHA 的格式 (batch_size, seq_len=1, input_dim)
        x = x.unsqueeze(1)  # 增加一个维度以适应 MHA 输入
        
        # 计算多头自注意力（MHA）
        attn_output, _ = self.mha(x, x, x)  # Q=K=V=x
        
        # 跳跃连接：将原始输入加到 MHA 输出上
        attn_output = attn_output.squeeze(1)  # 去掉多余的维度，形状变为 (batch_size, input_dim)
        attn_output = attn_output + x.squeeze(1)  # 加上原始输入

        # 使用 MLP 进行特征投影
        mlp_output = self.mlp(attn_output)
        output = mlp_output + attn_output
        
        return output


class FeatureExtractionModule(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers):
        super(FeatureExtractionModule, self).__init__()

        self.layers = nn.ModuleList([
            FeatureExtractionLayer(input_dim, num_heads
            ) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x