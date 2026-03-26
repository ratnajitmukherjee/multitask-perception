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

__all__ = ["get_cfg_defaults"]
