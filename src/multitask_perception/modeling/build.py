"""
Model building utilities.

This module provides factory functions for creating model instances
from configuration objects.
"""

from typing import Any

import torch.nn as nn

from multitask_perception.modeling.model import MultitaskPerceptionModel


def build_model(cfg: Any) -> nn.Module:
    """
    Build the multitask perception model from configuration.
    
    This is the main factory function for creating model instances.
    It handles all configuration validation and model instantiation.
    
    Args:
        cfg: Configuration object (yacs CfgNode) containing:
            - TASK.ENABLED: List of enabled tasks
            - MODEL.BACKBONE: Backbone configuration
            - MODEL.HEADS: Head configurations
            
    Returns:
        Initialized MultitaskPerceptionModel ready for training or inference
        
    Raises:
        ValueError: If configuration is invalid or missing required fields
        KeyError: If required configuration keys are missing
        
    Example:
        >>> from yacs.config import CfgNode
        >>> from multitask_perception.config import get_cfg_defaults
        >>> 
        >>> cfg = get_cfg_defaults()
        >>> cfg.merge_from_file('configs/experiments/kitti.yaml')
        >>> cfg.freeze()
        >>> 
        >>> model = build_model(cfg)
        >>> model.eval()
        >>> print(f"Enabled tasks: {model.enabled_tasks}")
    """
    # Validate configuration
    if not hasattr(cfg, "TASK") or not hasattr(cfg.TASK, "ENABLED"):
        raise ValueError(
            "Configuration must contain TASK.ENABLED field. "
            "Example: cfg.TASK.ENABLED = ['detection', 'segmentation']"
        )
    
    if not cfg.TASK.ENABLED:
        raise ValueError(
            "At least one task must be enabled. "
            "Set cfg.TASK.ENABLED to include 'detection', 'segmentation', or 'depth'"
        )
    
    # Validate task names
    valid_tasks = {"detection", "segmentation", "depth"}
    invalid_tasks = set(cfg.TASK.ENABLED) - valid_tasks
    if invalid_tasks:
        raise ValueError(
            f"Invalid task names: {invalid_tasks}. "
            f"Valid tasks are: {valid_tasks}"
        )
    
    # Create model
    model = MultitaskPerceptionModel(cfg)
    
    return model


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Number of parameters
        
    Example:
        >>> model = build_model(cfg)
        >>> trainable = count_parameters(model, trainable_only=True)
        >>> total = count_parameters(model, trainable_only=False)
        >>> print(f"Trainable: {trainable:,} / Total: {total:,}")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_summary(model: nn.Module) -> None:
    """
    Print a summary of the model architecture and parameters.
    
    Args:
        model: PyTorch model
        
    Example:
        >>> model = build_model(cfg)
        >>> print_model_summary(model)
        Model: MultitaskPerceptionModel
        Backbone: hardnet68
        Enabled tasks: ['detection', 'segmentation']
        Total parameters: 25,432,123
        Trainable parameters: 25,432,123
    """
    print(f"Model: {model.__class__.__name__}")
    
    if hasattr(model, "backbone"):
        print(f"Backbone: {model.backbone.__class__.__name__}")
    
    if hasattr(model, "enabled_tasks"):
        print(f"Enabled tasks: {model.enabled_tasks}")
    
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
