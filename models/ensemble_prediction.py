import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MLPBranch(nn.Module):
    """
    A simple MLP branch with variable layer depth.
    """
    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 3, hidden_dim: int = 256):
        super(MLPBranch, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class Classifier(nn.Module):
    """
    A simple classifier to choose which MLP branch to use.
    """
    def __init__(self, input_dim: int, num_classes: int = 4):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)  # (B, num_classes)
        return logits


class EnsemblePredictionModule(nn.Module):
    """
    Ensemble prediction module.
    
    When num_classes > 1:
        - Uses a classifier to choose one of multiple MLP branches
        - Returns (output, logits) where logits can be used for classification loss
    
    When num_classes == 1:
        - Degenerates to a single MLP (no classifier, no classification loss)
        - Returns (output, None)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_classes: int = 4,
        num_layers: int = 3,
        hidden_dim: int = 256,
    ):
        super(EnsemblePredictionModule, self).__init__()
        
        self.num_classes = num_classes
        self.output_dim = output_dim
        
        if num_classes == 1:
            # 退化为单个 MLP，不需要分类器
            self.mlp = MLPBranch(input_dim, output_dim, num_layers=num_layers, hidden_dim=hidden_dim)
            self.mlp_branches = None
            self.classifier = None
        else:
            # 多分支 + 分类器
            self.mlp = None
            self.mlp_branches = nn.ModuleList([
                MLPBranch(input_dim, output_dim, num_layers=num_layers, hidden_dim=hidden_dim)
                for _ in range(num_classes)
            ])
            self.classifier = Classifier(input_dim, num_classes)

    def forward(
        self, 
        x: torch.Tensor, 
        target_bin: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, D) input features
            target_bin: (B,) 训练时提供真实的分类标签（分桶索引），
                        用于 teacher forcing 选择分支。
                        如果为 None，则使用 classifier 的 argmax 选择分支。
        
        Returns:
            output: (B, output_dim) regression output
            logits: (B, num_classes) classification logits, or None if num_classes == 1
        """
        if self.num_classes == 1:
            # 单 MLP 模式：直接输出，不返回 logits
            output = self.mlp(x)  # (B, output_dim)
            return output, None
        
        # 多分支模式
        # 1. 获取分类 logits
        logits = self.classifier(x)  # (B, num_classes)
        
        # 2. 选择分支
        if target_bin is not None:
            # 训练时：使用真实的分类标签（teacher forcing）
            class_idx = target_bin  # (B,)
        else:
            # 推理时：使用 classifier 的 argmax
            class_idx = torch.argmax(logits, dim=-1)  # (B,)
        
        # 确保 class_idx 在有效范围内
        class_idx = class_idx.clamp(0, self.num_classes - 1)
        one_hot = F.one_hot(class_idx, num_classes=self.num_classes).float()  # (B, num_classes)
        
        # 3. 计算每个分支的输出
        outputs = torch.stack([mlp(x) for mlp in self.mlp_branches], dim=1)  # (B, num_classes, output_dim)
        
        # 4. 加权求和（hard selection 时等同于选择对应分支）
        # one_hot: (B, num_classes) -> (B, num_classes, 1)
        output = (one_hot.unsqueeze(-1) * outputs).sum(dim=1)  # (B, output_dim)
        
        return output, logits
