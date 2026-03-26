"""
Depth estimation heads for monocular depth prediction.

Available architectures:
    - SimpleDecoder: Basic depth decoder (coming soon)
    - MonoDepth: MonoDepth-style decoder (planned)

Note: Depth heads will be implemented in the next phase.
"""

from typing import Any

import torch.nn as nn


def build_depth_head(cfg: Any) -> nn.Module:
    """
    Build depth estimation head from configuration.
    
    Args:
        cfg: Configuration object containing MODEL.HEADS.DEPTH settings
        
    Returns:
        Initialized depth head
        
    Raises:
        NotImplementedError: Depth heads not yet implemented
        
    Note:
        This will be implemented in the next development phase.
        For now, depth estimation is not available.
    """
    raise NotImplementedError(
        "Depth estimation heads not yet implemented. "
        "This feature is planned for the next phase. "
        "Please disable 'depth' from cfg.TASK.ENABLED for now."
    )


__all__ = ["build_depth_head"]
