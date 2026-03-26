"""
HarDNet (Harmonic Dense Network) backbone implementation.

Reference:
    HarDNet: A Low Memory Traffic Network
    https://arxiv.org/abs/1909.00948
"""

from typing import Any

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """Basic convolutional layer with BatchNorm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class HarDBlock(nn.Module):
    """HarDNet dense block with harmonic connections."""
    
    def __init__(self, in_channels: int, growth_rate: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.links = []
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            outch, inch = growth_rate, 0
            for j in range(10):
                dv = 2 ** j
                if (i + 1) % dv == 0:
                    inch += growth_rate if j > 0 else in_channels
                    self.links.append(i - dv + 1)
            
            self.layers.append(ConvLayer(inch, outch))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layers_ = [x]
        for layer in self.layers:
            link = []
            for i in self.links:
                if i >= 0:
                    link.append(layers_[i])
            link = torch.cat(link, 1) if len(link) > 0 else layers_[0]
            out = layer(link)
            layers_.append(out)
        
        return torch.cat(layers_[1:], 1)


class HarDNet(nn.Module):
    """
    HarDNet backbone network.
    
    Args:
        block_configs: List of (in_ch, growth_rate, num_layers) for each block
        pretrained: Whether to load ImageNet pretrained weights
    """
    
    def __init__(self, block_configs: list[tuple[int, int, int]], pretrained: bool = False) -> None:
        super().__init__()
        
        # Initial conv
        self.conv1 = ConvLayer(3, 32, kernel_size=3, stride=2)
        
        # Build blocks
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        in_ch = 32
        for i, (_, growth_rate, num_layers) in enumerate(block_configs):
            self.blocks.append(HarDBlock(in_ch, growth_rate, num_layers))
            out_ch = in_ch + growth_rate * num_layers
            
            if i < len(block_configs) - 1:
                self.transitions.append(ConvLayer(out_ch, out_ch // 2, kernel_size=1))
                in_ch = out_ch // 2
            else:
                in_ch = out_ch
        
        self.out_channels = [32, in_ch]
        
        if pretrained:
            print("Warning: HarDNet pretrained weights not implemented. Using random initialization.")
    
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
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        
        features.append(x)
        return features


class HarDNet68(HarDNet):
    """HarDNet-68 backbone (17.6M parameters)."""
    
    def __init__(self, pretrained: bool = False) -> None:
        block_configs = [
            (32, 32, 4),   # Block 1
            (64, 32, 8),   # Block 2
            (128, 32, 16), # Block 3
            (256, 32, 16), # Block 4
        ]
        super().__init__(block_configs, pretrained)


class HarDNet85(HarDNet):
    """HarDNet-85 backbone (36.7M parameters)."""
    
    def __init__(self, pretrained: bool = False) -> None:
        block_configs = [
            (32, 48, 4),   # Block 1
            (96, 48, 8),   # Block 2
            (192, 48, 16), # Block 3
            (384, 48, 16), # Block 4
        ]
        super().__init__(block_configs, pretrained)
