"""
Multitask Perception for Autonomous Driving.

A unified PyTorch framework for multitask perception combining:
- Backbone (HardNet, VovNet, MobileNetV3) 
- Object Detection (NanoDet, CenterNet)
- Semantic Segmentation (SegFormer, DeepLabV3, ESPNetV2)
- Monocular Depth Estimation

Core module for multitask perception.

Contains:
    - config: Configuration management
    - data: Dataset loaders and transforms
    - modeling: Model architectures, backbones, and heads
    - engine: Training and evaluation engines
    - solver: Optimizers and learning rate schedulers
    - utils: Utility functions

Author: Ratnajit Mukherjee
License: MIT
"""

# from multitask_perception.__version__ import __version__


__all__ = ["__version__", "config", "data", "modeling", "engine", "solver", "utils"]
