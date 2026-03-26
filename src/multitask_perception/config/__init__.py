"""
Configuration management using YACS.

This module provides default configuration and utilities for
managing experiment configurations.

Example:
    >>> from multitask_perception.config import get_cfg_defaults
    >>> cfg = get_cfg_defaults()
    >>> cfg.merge_from_file('configs/experiment.yaml')
    >>> cfg.freeze()
"""

from multitask_perception.config.defaults import get_cfg_defaults
from multitask_perception.config.head_configs import sub_cfg_dict

__all__ = ["get_cfg_defaults", "sub_cfg_dict"]
