"""
VoVNet (One-Shot Aggregation) backbone implementation.

Reference:
    An Energy and GPU-Computation Efficient Backbone Network
    https://arxiv.org/abs/1904.09730
"""

from typing import Any

import torch
import torch.nn as nn


class OSAModule(nn.Module):
    """One-Shot Aggregation module."""
    
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        num_layers: int = 5,
    ) -> None:
        super().__init__()
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(self._make_layer(in_channels, mid_channels))
        
        # Middle layers
        for _ in range(num_layers - 1):
            self.layers.append(self._make_layer(mid_channels, mid_channels))
        
        # Concat all layers + input
        concat_channels = in_channels + mid_channels * num_layers
        self.concat_conv = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def _make_layer(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [x]
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        
        x = torch.cat(outputs, dim=1)
        x = self.concat_conv(x)
        return x


class VoVNet(nn.Module):
    """
    VoVNet backbone network.
    
    Args:
        stage_configs: List of (in_ch, mid_ch, out_ch, num_blocks) for each stage
        pretrained: Whether to load ImageNet pretrained weights
    """
    
    def __init__(
        self,
        stage_configs: list[tuple[int, int, int, int]],
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Stages
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        for i, (in_ch, mid_ch, out_ch, num_blocks) in enumerate(stage_configs):
            # Build stage
            stage = nn.Sequential(
                *[OSAModule(in_ch if j == 0 else out_ch, mid_ch, out_ch)
                  for j in range(num_blocks)]
            )
            self.stages.append(stage)
            
            # Transition (downsampling)
            if i < len(stage_configs) - 1:
                self.transitions.append(nn.MaxPool2d(3, stride=2, padding=1))
        
        self.out_channels = [64] + [cfg[2] for cfg in stage_configs]
        
        if pretrained:
            print("Warning: VoVNet pretrained weights not implemented. Using random initialization.")
    
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass returning multi-scale features.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            List of feature maps at different scales
        """
        features = []
        
        x = self.stem(x)
        features.append(x)
        
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        
        return features


class VoVNet27Slim(VoVNet):
    """VoVNet-27-slim backbone (3.5M parameters) - Fast."""
    
    def __init__(self, pretrained: bool = False) -> None:
        stage_configs = [
            (64, 64, 128, 1),   # Stage 1
            (128, 80, 256, 1),  # Stage 2
            (256, 96, 384, 1),  # Stage 3
            (384, 112, 512, 1), # Stage 4
        ]
        super().__init__(stage_configs, pretrained)


class VoVNet39(VoVNet):
    """VoVNet-39 backbone (22.6M parameters) - Balanced."""
    
    def __init__(self, pretrained: bool = False) -> None:
        stage_configs = [
            (64, 128, 256, 1),   # Stage 1
            (256, 160, 512, 1),  # Stage 2
            (512, 192, 768, 2),  # Stage 3
            (768, 224, 1024, 2), # Stage 4
        ]
        super().__init__(stage_configs, pretrained)
