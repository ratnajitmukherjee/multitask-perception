"""
Task-specific prediction heads.

This module provides detection, segmentation, and depth estimation heads
for the multitask perception system.

Available Heads:
    Detection:
        - NanoDet: Anchor-free, fast, accurate
        - CenterNet: Keypoint-based detection
    
    Segmentation:
        - SegFormer-B0: Modern transformer, 3.7M params
        - DeepLabV3: Classic, high accuracy, 40M+ params
        - ESPNetV2: Ultra-efficient, 0.8M params
    
    Depth:
        - SimpleDecoder: Basic depth decoder (coming soon)

Example:
    >>> from multitask_perception.modeling.heads import build_head
    >>> det_head = build_head(cfg, task='detection')
    >>> seg_head = build_head(cfg, task='segmentation')
"""

from typing import Any

import torch.nn as nn


def build_head(cfg: Any, task: str) -> nn.Module:
    """
    Build task-specific head from configuration.
    
    Args:
        cfg: Configuration object
        task: Task name ('detection', 'segmentation', or 'depth')
        
    Returns:
        Initialized head network
        
    Raises:
        ValueError: If task is not recognized or head name is invalid
        NotImplementedError: If head is not yet implemented
        
    Example:
        >>> cfg.MODEL.HEADS.DETECTION.NAME = 'NanoDet'
        >>> det_head = build_head(cfg, task='detection')
    """
    if task == "detection":
        from multitask_perception.modeling.heads.detection import build_detection_head
        return build_detection_head(cfg)
    
    elif task == "segmentation":
        from multitask_perception.modeling.heads.segmentation import build_segmentation_head
        return build_segmentation_head(cfg)
    
    elif task == "depth":
        from multitask_perception.modeling.heads.depth import build_depth_head
        return build_depth_head(cfg)
    
    else:
        raise ValueError(
            f"Unknown task: {task}. "
            f"Valid tasks are: 'detection', 'segmentation', 'depth'"
        )


__all__ = ["build_head"]
