"""
Detection heads for object detection.

Available architectures:
    - NanoDet: Lightweight anchor-free detector
    - CenterNet: Keypoint-based detector
"""

from typing import Any

import torch.nn as nn


def build_detection_head(cfg: Any) -> nn.Module:
    """
    Build detection head from configuration.
    
    Args:
        cfg: Configuration object containing MODEL.HEADS.DETECTION settings
        
    Returns:
        Initialized detection head
        
    Raises:
        ValueError: If detection head name is not recognized
        NotImplementedError: If head is not yet implemented
    """
    head_name = cfg.MODEL.HEADS.DETECTION.NAME.lower()
    
    if head_name == "nanodet":
        from multitask_perception.modeling.heads.detection.nanodet import NanoDetHead
        return NanoDetHead(cfg)
    
    elif head_name == "centernet":
        from multitask_perception.modeling.heads.detection.centernet import CenterNetHead
        return CenterNetHead(cfg)
    
    else:
        raise ValueError(
            f"Unknown detection head: {head_name}. "
            f"Available: 'nanodet', 'centernet'"
        )


__all__ = ["build_detection_head"]
