import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.num_layers = num_layers

        # 第 1 层：input_dim -> output_dim
        self.lstm1 = nn.LSTM(input_dim, output_dim, num_layers=1, batch_first=True)

        # 后续层：output_dim -> output_dim
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(output_dim, output_dim, num_layers=1, batch_first=True)
            for _ in range(1, num_layers)
        ])

        self.fc = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (B, Lmax, input_dim) padded sequence batch
          seq_len: (B,) true lengths (<= Lmax)

        Returns:
          emb: (B, output_dim)
        """
        if x.ndim != 3:
            raise ValueError(f"x must be 3D (B,L,D), got {tuple(x.shape)}")
        if seq_len.ndim != 1:
            raise ValueError(f"seq_len must be 1D (B,), got {tuple(seq_len.shape)}")

        B, Lmax, _ = x.shape

        # clamp lengths into valid range to avoid runtime errors
        seq_len = seq_len.to(torch.long)
        seq_len = torch.clamp(seq_len, min=1, max=Lmax)

        # 为 pack_padded_sequence 需要：lengths 在 CPU 上，且通常要求降序（我们用 enforce_sorted=False 免排序）
        lengths_cpu = seq_len.detach().cpu()

        # ---- Layer 1 ----
        packed = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm1(packed)
        # h_n: (num_layers=1, B, output_dim) => 取最后层
        emb = h_n[-1]  # (B, output_dim)

        # 为后续层准备：需要把 packed_out 再解包成 padded，再 pack 进入下一层
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=Lmax)

        # ---- Subsequent layers ----
        for lstm in self.lstm_layers:
            packed = pack_padded_sequence(out, lengths_cpu, batch_first=True, enforce_sorted=False)
            packed_out, (h_n, c_n) = lstm(packed)
            emb = h_n[-1]  # (B, output_dim)
            out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=Lmax)

        # ---- FC projection ----
        emb = self.fc(emb)  # (B, output_dim)
        return emb
