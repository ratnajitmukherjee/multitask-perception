"""
MobileNetV3 backbone implementation.

Reference:
    Searching for MobileNetV3
    https://arxiv.org/abs/1905.02244
"""

from typing import Any

import torch
import torch.nn as nn


class HSwish(nn.Module):
    """Hard Swish activation function."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.clamp(x + 3, 0, 6) / 6


class HSigmoid(nn.Module):
    """Hard Sigmoid activation function."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x + 3, 0, 6) / 6


class SEModule(nn.Module):
    """Squeeze-and-Excitation module."""
    
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            HSigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MobileBottleneck(nn.Module):
    """MobileNetV3 inverted residual block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        use_se: bool,
        use_hs: bool,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
        hidden_dim = in_channels * expand_ratio
        activation = HSwish if use_hs else nn.ReLU
        
        layers = []
        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace=True),
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size,
                stride=stride, padding=kernel_size // 2,
                groups=hidden_dim, bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            activation(inplace=True),
        ])
        
        # SE
        if use_se:
            layers.append(SEModule(hidden_dim))
        
        # Project
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    """
    MobileNetV3 backbone network.
    
    Args:
        block_configs: List of block configurations
        pretrained: Whether to load ImageNet pretrained weights
    """
    
    def __init__(
        self,
        block_configs: list[tuple[int, int, int, int, int, bool, bool]],
        last_channels: int,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        
        # First conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HSwish(inplace=True),
        )
        
        # Blocks
        self.blocks = nn.ModuleList()
        in_channels = 16
        
        for exp_ratio, out_ch, kernel, stride, use_se, use_hs, _ in block_configs:
            self.blocks.append(
                MobileBottleneck(
                    in_channels, out_ch, kernel, stride,
                    exp_ratio, use_se, use_hs
                )
            )
            in_channels = out_ch
        
        # Last conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, last_channels, 1, bias=False),
            nn.BatchNorm2d(last_channels),
            HSwish(inplace=True),
        )
        
        self.out_channels = [16, last_channels]
        
        if pretrained:
            print("Warning: MobileNetV3 pretrained weights not implemented. Using random initialization.")
    
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass returning multi-scale features.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            List of feature maps at different scales
        """
        features = []
        
        x = self.conv1(x)
        features.append(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.conv2(x)
        features.append(x)
        
        return features


class MobileNetV3Small(MobileNetV3):
    """MobileNetV3-Small backbone (2.5M parameters) - Ultra fast."""
    
    def __init__(self, pretrained: bool = False) -> None:
        # exp_ratio, out_ch, kernel, stride, use_se, use_hs, stage
        block_configs = [
            (1, 16, 3, 2, True, False, 1),
            (4, 24, 3, 2, False, False, 2),
            (3, 24, 3, 1, False, False, 2),
            (3, 40, 5, 2, True, True, 3),
            (3, 40, 5, 1, True, True, 3),
            (3, 40, 5, 1, True, True, 3),
            (6, 48, 5, 1, True, True, 4),
            (6, 48, 5, 1, True, True, 4),
            (6, 96, 5, 2, True, True, 5),
            (6, 96, 5, 1, True, True, 5),
            (6, 96, 5, 1, True, True, 5),
        ]
        super().__init__(block_configs, last_channels=576, pretrained=pretrained)


class MobileNetV3Large(MobileNetV3):
    """MobileNetV3-Large backbone (5.4M parameters) - Fast."""
    
    def __init__(self, pretrained: bool = False) -> None:
        # exp_ratio, out_ch, kernel, stride, use_se, use_hs, stage
        block_configs = [
            (1, 16, 3, 1, False, False, 1),
            (4, 24, 3, 2, False, False, 2),
            (3, 24, 3, 1, False, False, 2),
            (3, 40, 5, 2, True, False, 3),
            (3, 40, 5, 1, True, False, 3),
            (3, 40, 5, 1, True, False, 3),
            (6, 80, 3, 2, False, True, 4),
            (2.5, 80, 3, 1, False, True, 4),
            (2.3, 80, 3, 1, False, True, 4),
            (2.3, 80, 3, 1, False, True, 4),
            (6, 112, 3, 1, True, True, 5),
            (6, 112, 3, 1, True, True, 5),
            (6, 160, 5, 2, True, True, 6),
            (6, 160, 5, 1, True, True, 6),
            (6, 160, 5, 1, True, True, 6),
        ]
        super().__init__(block_configs, last_channels=960, pretrained=pretrained)
