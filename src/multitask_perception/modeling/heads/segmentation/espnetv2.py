"""
ESPNetV2 segmentation head (placeholder implementation).

This is a minimal placeholder to make the structure work.
Replace with your actual ESPNetV2 implementation.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ESPNetV2Head(nn.Module):
    """
    ESPNetV2 segmentation head - placeholder implementation.
    
    TODO: Copy the full ESPNetV2 implementation from your original codebase:
        - od/modeling/head/segmentation/espnetv2_seg.py
    
    Args:
        cfg: Configuration object
    """
    
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.MODEL.HEADS.SEGMENTATION.get("NUM_CLASSES", 19)
        
        # Placeholder lightweight decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.num_classes, 1),
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        print("=" * 60)
        print("WARNING: Using placeholder ESPNetV2 head!")
        print("Replace with actual implementation from your codebase:")
        print("  Copy from: od/modeling/head/segmentation/espnetv2_seg.py")
        print("=" * 60)
    
    def forward(
        self,
        features: list[torch.Tensor],
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward pass."""
        x = features[-1] if isinstance(features, list) else features
        pred = self.decoder(x)
        pred = F.interpolate(pred, scale_factor=4, mode='bilinear', align_corners=False)
        
        if self.training and targets is not None:
            loss = self.criterion(pred, targets.long())
            return pred, loss
        else:
            return pred


__all__ = ["ESPNetV2Head"]
