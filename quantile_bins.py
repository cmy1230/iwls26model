import numpy as np
import torch
from typing import Optional, Union
import json
import os


class QuantileBinManager:
    """
    管理每个任务的分位数边界，用于将连续值划分到不同的分桶中。
    
    使用分位数划分可以确保每个分桶中的样本数量大致相等，
    解决数据分布不均匀（如大量0-1和1-100分布不均）的问题。
    """
    
    def __init__(self, num_tasks: int, num_bins: int):
        """
        Args:
            num_tasks: 任务数量
            num_bins: 分桶数量（分支数量）
        """
        self.num_tasks = num_tasks
        self.num_bins = num_bins
        self.boundaries: Optional[np.ndarray] = None  # (T, num_bins-1) 每个任务的边界点
        self._fitted = False
    
    def fit(self, labels: np.ndarray) -> "QuantileBinManager":
        """
        从训练数据计算分位数边界。
        
        Args:
            labels: (N, T) 所有训练样本的标签，N为样本数，T为任务数
            
        Returns:
            self
        """
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        
        N, T = labels.shape
        if T != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} tasks, got {T}")
        
        # 计算分位数点: 例如 4 个 bins 需要 [25, 50, 75] 三个边界
        quantile_points = np.linspace(0, 100, self.num_bins + 1)[1:-1]  # 排除 0 和 100
        
        # 对每个任务计算分位数边界
        # np.percentile(labels, quantiles, axis=0) 返回 (len(quantiles), T)
        self.boundaries = np.percentile(labels, quantile_points, axis=0).T  # (T, num_bins-1)
        
        # 确保边界严格递增（处理相同值的情况）
        for t in range(T):
            for i in range(1, self.num_bins - 1):
                if self.boundaries[t, i] <= self.boundaries[t, i - 1]:
                    # 添加微小偏移确保严格递增
                    self.boundaries[t, i] = self.boundaries[t, i - 1] + 1e-6
        
        self._fitted = True
        return self
    
    def get_bin_indices(self, labels: torch.Tensor) -> torch.Tensor:
        """
        将标签转换为分类索引 (0 ~ num_bins-1)。
        
        Args:
            labels: (B, T) 或 (B,) 标签张量
            
        Returns:
            bin_indices: (B, T) 或 (B,) 分桶索引，取值范围 [0, num_bins-1]
        """
        if not self._fitted:
            raise RuntimeError("QuantileBinManager must be fitted before use")
        
        device = labels.device
        dtype = labels.dtype
        
        # 处理一维输入
        squeeze_output = False
        if labels.ndim == 1:
            labels = labels.unsqueeze(-1)
            squeeze_output = True
        
        B, T = labels.shape
        if T != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} tasks, got {T}")
        
        # 转换边界到 torch tensor
        boundaries_tensor = torch.tensor(self.boundaries, device=device, dtype=dtype)  # (T, num_bins-1)
        
        # 对每个任务进行分桶
        bin_indices = torch.zeros(B, T, dtype=torch.long, device=device)
        
        for t in range(T):
            # boundaries_tensor[t] 是 (num_bins-1,) 的边界点
            # bucketize 返回每个值应该插入的位置，即分桶索引
            bin_indices[:, t] = torch.bucketize(
                labels[:, t].contiguous(), 
                boundaries_tensor[t].contiguous()
            )
        
        if squeeze_output:
            bin_indices = bin_indices.squeeze(-1)
        
        return bin_indices
    
    def get_bin_indices_numpy(self, labels: np.ndarray) -> np.ndarray:
        """
        NumPy 版本的分桶索引计算。
        
        Args:
            labels: (N, T) 或 (N,) 标签数组
            
        Returns:
            bin_indices: (N, T) 或 (N,) 分桶索引
        """
        if not self._fitted:
            raise RuntimeError("QuantileBinManager must be fitted before use")
        
        squeeze_output = False
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
            squeeze_output = True
        
        N, T = labels.shape
        if T != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} tasks, got {T}")
        
        bin_indices = np.zeros((N, T), dtype=np.int64)
        
        for t in range(T):
            # np.searchsorted 等价于 torch.bucketize
            bin_indices[:, t] = np.searchsorted(self.boundaries[t], labels[:, t])
        
        if squeeze_output:
            bin_indices = bin_indices.squeeze(-1)
        
        return bin_indices
    
    def save(self, path: str) -> None:
        """
        保存分位数边界到文件。
        
        Args:
            path: 保存路径（.json 或 .npz）
        """
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted QuantileBinManager")
        
        data = {
            "num_tasks": self.num_tasks,
            "num_bins": self.num_bins,
            "boundaries": self.boundaries.tolist(),
        }
        
        if path.endswith(".json"):
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            np.savez(path, **{k: np.array(v) for k, v in data.items()})
    
    @classmethod
    def load(cls, path: str) -> "QuantileBinManager":
        """
        从文件加载分位数边界。
        
        Args:
            path: 加载路径
            
        Returns:
            QuantileBinManager 实例
        """
        if path.endswith(".json"):
            with open(path, "r") as f:
                data = json.load(f)
            num_tasks = data["num_tasks"]
            num_bins = data["num_bins"]
            boundaries = np.array(data["boundaries"])
        else:
            loaded = np.load(path, allow_pickle=True)
            num_tasks = int(loaded["num_tasks"])
            num_bins = int(loaded["num_bins"])
            boundaries = loaded["boundaries"]
        
        manager = cls(num_tasks=num_tasks, num_bins=num_bins)
        manager.boundaries = boundaries
        manager._fitted = True
        return manager
    
    def get_bin_statistics(self, labels: np.ndarray) -> dict:
        """
        获取每个分桶的统计信息（用于调试和可视化）。
        
        Args:
            labels: (N, T) 标签数组
            
        Returns:
            统计信息字典
        """
        if not self._fitted:
            raise RuntimeError("QuantileBinManager must be fitted before use")
        
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        
        bin_indices = self.get_bin_indices_numpy(labels)
        
        stats = {}
        for t in range(self.num_tasks):
            task_stats = {}
            for b in range(self.num_bins):
                mask = bin_indices[:, t] == b
                count = mask.sum()
                if count > 0:
                    values = labels[mask, t]
                    task_stats[f"bin_{b}"] = {
                        "count": int(count),
                        "percentage": float(count / len(labels) * 100),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "mean": float(values.mean()),
                    }
                else:
                    task_stats[f"bin_{b}"] = {"count": 0, "percentage": 0.0}
            
            # 添加边界信息
            task_stats["boundaries"] = self.boundaries[t].tolist()
            stats[f"task_{t}"] = task_stats
        
        return stats
    
    @property
    def is_fitted(self) -> bool:
        return self._fitted
    
    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return f"QuantileBinManager(num_tasks={self.num_tasks}, num_bins={self.num_bins}, {status})"
