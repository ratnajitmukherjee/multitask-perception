"""
SegFormer segmentation head (placeholder implementation).

This is a minimal placeholder to make the structure work.
Replace with actual SegFormer or use segmentation_models_pytorch.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerHead(nn.Module):
    """
    SegFormer segmentation head - placeholder implementation.
    
    TODO: Either:
    1. Use segmentation_models_pytorch library (recommended):
       import segmentation_models_pytorch as smp
       model = smp.create_model('segformer', encoder_name='mit_b0', ...)
    
    2. Or implement full SegFormer from scratch
    
    Args:
        cfg: Configuration object
    """
    
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.MODEL.HEADS.SEGMENTATION.get("NUM_CLASSES", 19)
        
        # Placeholder simple decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.num_classes, 1),
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        print("=" * 60)
        print("WARNING: Using placeholder SegFormer head!")
        print("Recommended: Use segmentation_models_pytorch library")
        print("  pip install segmentation-models-pytorch")
        print("=" * 60)
    
    def forward(
        self,
        features: list[torch.Tensor],
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: List of feature maps from backbone
            targets: Ground truth segmentation masks [B, H, W] (training only)
            
        Returns:
            If training: (predictions [B, C, H, W], loss)
            If inference: predictions [B, C, H, W]
        """
        # Use last feature map
        x = features[-1] if isinstance(features, list) else features
        
        # Decode
        pred = self.decoder(x)
        
        # Upsample to input size
        pred = F.interpolate(pred, scale_factor=4, mode='bilinear', align_corners=False)
        
        if self.training and targets is not None:
            # Compute loss
            loss = self.criterion(pred, targets.long())
            return pred, loss
        else:
            return pred


__all__ = ["SegFormerHead"]
