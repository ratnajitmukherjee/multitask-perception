"""
Backbone networks for feature extraction.

This module provides efficient backbone architectures for the multitask
perception system. All backbones follow a consistent interface.

Available Backbones:
    - HarDNet-68, HarDNet-85: Harmonic Dense Networks
    - VoVNet-27-slim, VoVNet-39: One-Shot Aggregation Networks
    - MobileNetV3-Small, MobileNetV3-Large: Efficient mobile architectures

Example:
    >>> from multitask_perception.modeling.backbones import build_backbone
    >>> backbone = build_backbone(cfg)
    >>> features = backbone(images)
"""

from typing import Any

import torch.nn as nn

from multitask_perception.modeling.backbones.hardnet import HarDNet68, HarDNet85
from multitask_perception.modeling.backbones.vovnet import VoVNet27Slim, VoVNet39
from multitask_perception.modeling.backbones.mobilenetv3 import (
    MobileNetV3Small,
    MobileNetV3Large,
)


# Registry of available backbones
BACKBONE_REGISTRY = {
    "hardnet68": HarDNet68,
    "hardnet85": HarDNet85,
    "vovnet27_slim": VoVNet27Slim,
    "vovnet39": VoVNet39,
    "mobilenet_v3_small": MobileNetV3Small,
    "mobilenet_v3_large": MobileNetV3Large,
}


def build_backbone(cfg: Any) -> nn.Module:
    """
    Build backbone network from configuration.
    
    Args:
        cfg: Configuration object containing:
            - MODEL.BACKBONE.NAME: Backbone architecture name
            - MODEL.BACKBONE.PRETRAINED: Whether to load pretrained weights
            - MODEL.BACKBONE.FREEZE: Whether to freeze backbone parameters
            
    Returns:
        Initialized backbone network
        
    Raises:
        ValueError: If backbone name is not recognized
        
    Example:
        >>> cfg.MODEL.BACKBONE.NAME = 'hardnet68'
        >>> cfg.MODEL.BACKBONE.PRETRAINED = True
        >>> backbone = build_backbone(cfg)
    """
    backbone_name = cfg.MODEL.BACKBONE.NAME.lower()
    
    if backbone_name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone: {backbone_name}. "
            f"Available backbones: {list(BACKBONE_REGISTRY.keys())}"
        )
    
    # Get backbone class
    backbone_cls = BACKBONE_REGISTRY[backbone_name]
    
    # Get pretrained setting
    pretrained = cfg.MODEL.BACKBONE.get("PRETRAINED", False)
    
    # Build backbone
    backbone = backbone_cls(pretrained=pretrained)
    
    # Freeze backbone if requested
    if cfg.MODEL.BACKBONE.get("FREEZE", False):
        for param in backbone.parameters():
            param.requires_grad = False
        print(f"Backbone {backbone_name} frozen (not trainable)")
    
    return backbone


__all__ = [
    "build_backbone",
    "HarDNet68",
    "HarDNet85",
    "VoVNet27Slim",
    "VoVNet39",
    "MobileNetV3Small",
    "MobileNetV3Large",
    "BACKBONE_REGISTRY",
]
