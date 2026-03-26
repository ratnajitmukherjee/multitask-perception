"""
Modeling module for multitask perception.

Contains:
    - backbones: Feature extraction networks
    - necks: Feature pyramid networks
    - heads: Task-specific prediction heads
    - losses: Loss functions
    - layers: Custom layers and modules
    - temporal: Temporal modules for video (future)
    - tracking: Object tracking modules (future)
"""

from multitask_perception.modeling.model import MultitaskPerceptionModel
from multitask_perception.modeling.build import build_model

__all__ = [
    "MultitaskPerceptionModel",
    "build_model",
]