import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        # 第一层将 input_dim 扩展到 output_dim
        self.lstm1 = nn.LSTM(input_dim, output_dim, num_layers=1, batch_first=True)
        
        # 后续的层维度保持 output_dim
        self.lstm_layers = nn.ModuleList()
        for _ in range(1, num_layers):
            self.lstm_layers.append(nn.LSTM(output_dim, output_dim, num_layers=1, batch_first=True))
        
        # 一个全连接层（可选）
        self.fc = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        # x 形状为 (batch_size, seq_len, input_dim)
        
        # 第1层 LSTM
        out, _ = self.lstm1(x)
        
        # 后续 LSTM 层
        for lstm in self.lstm_layers:
            out, _ = lstm(out)
        
        # 使用全连接层进行输出
        out = self.fc(out[:, -1, :])  # 取序列的最后一个时间步的输出
        
        return out