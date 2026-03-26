"""
Segmentation heads for semantic segmentation.

Available architectures:
    - SegFormer-B0: Modern transformer-based
    - DeepLabV3: Classic with ASPP
    - ESPNetV2: Ultra-efficient
"""

from typing import Any

import torch.nn as nn


def build_segmentation_head(cfg: Any) -> nn.Module:
    """
    Build segmentation head from configuration.
    
    Args:
        cfg: Configuration object containing MODEL.HEADS.SEGMENTATION settings
        
    Returns:
        Initialized segmentation head
        
    Raises:
        ValueError: If segmentation head name is not recognized
        NotImplementedError: If head is not yet implemented
    """
    head_name = cfg.MODEL.HEADS.SEGMENTATION.NAME.lower()
    
    if head_name == "segformer" or head_name == "segformer_b0":
        from multitask_perception.modeling.heads.segmentation.segformer import SegFormerHead
        return SegFormerHead(cfg)
    
    elif head_name == "deeplabv3":
        from multitask_perception.modeling.heads.segmentation.deeplabv3 import DeepLabV3Head
        return DeepLabV3Head(cfg)
    
    elif head_name == "espnetv2":
        from multitask_perception.modeling.heads.segmentation.espnetv2 import ESPNetV2Head
        return ESPNetV2Head(cfg)
    
    else:
        raise ValueError(
            f"Unknown segmentation head: {head_name}. "
            f"Available: 'segformer', 'deeplabv3', 'espnetv2'"
        )


__all__ = ["build_segmentation_head"]
