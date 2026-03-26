"""
Main multitask perception model.

This module contains the core MultitaskPerceptionModel class that combines
object detection, semantic segmentation, and depth estimation in a unified
architecture.
"""

from typing import Any

import torch
import torch.nn as nn

from multitask_perception.modeling.backbones import build_backbone
from multitask_perception.modeling.heads import build_head


class MultitaskPerceptionModel(nn.Module):
    """
    Unified multitask perception model for autonomous driving.
    
    This model supports simultaneous object detection, semantic segmentation,
    and monocular depth estimation through a shared backbone architecture.
    Designed to be video-ready with hooks for temporal modules.
    
    Architecture:
        Input Image → Backbone → Neck (optional) → Task Heads → Outputs
        
    Attributes:
        cfg: Configuration object containing all model settings
        backbone: Feature extraction network (HarDNet/VoVNet/MobileNetV3)
        neck: Optional feature pyramid network (currently disabled)
        use_temporal: Whether temporal processing is enabled (for video)
        enabled_tasks: List of active tasks ['detection', 'segmentation', 'depth']
        heads: Dictionary of task-specific heads
        
    Example:
        >>> from yacs.config import CfgNode
        >>> cfg = CfgNode()
        >>> cfg.TASK = CfgNode()
        >>> cfg.TASK.ENABLED = ['detection', 'segmentation']
        >>> model = MultitaskPerceptionModel(cfg)
        >>> images = torch.randn(2, 3, 640, 640)
        >>> outputs, losses, _ = model(images)
    """
    
    def __init__(self, cfg: Any) -> None:
        """
        Initialize the multitask perception model.
        
        Args:
            cfg: Configuration object (yacs CfgNode) containing:
                - TASK.ENABLED: List of enabled tasks
                - MODEL.BACKBONE: Backbone configuration
                - MODEL.NECK: Neck configuration (optional)
                - MODEL.TEMPORAL: Temporal module configuration
                - MODEL.HEADS: Task head configurations
        """
        super().__init__()
        self.cfg = cfg
        self.enabled_tasks = cfg.TASK.ENABLED
        
        # Build shared backbone
        self.backbone = build_backbone(cfg)
        
        # Optional neck (FPN, PAN, etc.) - currently disabled
        self.neck = None
        # TODO: Add neck support
        # if cfg.MODEL.NECK.ENABLED:
        #     self.neck = build_neck(cfg)
        
        # Temporal module for video (future feature)
        self.use_temporal = cfg.MODEL.get("TEMPORAL", {}).get("ENABLED", False)
        if self.use_temporal:
            # TODO: Implement temporal modules in 2-3 months
            raise NotImplementedError(
                "Temporal modules not yet implemented. "
                "Set cfg.MODEL.TEMPORAL.ENABLED = False for image-based inference."
            )
        
        # Build task-specific heads
        self.heads = nn.ModuleDict()
        
        if "detection" in self.enabled_tasks:
            self.heads["detection"] = build_head(cfg, task="detection")
        
        if "segmentation" in self.enabled_tasks:
            self.heads["segmentation"] = build_head(cfg, task="segmentation")
        
        if "depth" in self.enabled_tasks:
            self.heads["depth"] = build_head(cfg, task="depth")
    
    def forward(
        self,
        images: torch.Tensor,
        targets: dict[str, Any] | None = None,
        temporal_state: dict[str, Any] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any] | None]:
        """
        Forward pass supporting both image and video inputs.
        
        Args:
            images: Input tensor
                - Images: [B, 3, H, W]
                - Video (future): [B, T, 3, H, W] where T is number of frames
            targets: Ground truth targets (training only). Dictionary containing:
                - 'boxes': Detection boxes [B, N, 4] (optional)
                - 'labels': Detection labels [B, N] (optional)
                - 'seg_mask': Segmentation masks [B, H, W] (optional)
                - 'depth': Depth maps [B, H, W] (optional)
            temporal_state: Optional temporal state for video inference (future)
                None for images, state dict for video sequences
                
        Returns:
            outputs: Task predictions dictionary with keys:
                - 'detection': Detection results (if enabled)
                - 'segmentation': Segmentation masks (if enabled)
                - 'depth': Depth maps (if enabled)
            losses: Task losses dictionary (training only, empty during inference):
                - 'det_loss': Detection loss (if enabled)
                - 'seg_loss': Segmentation loss (if enabled)
                - 'depth_loss': Depth loss (if enabled)
            new_temporal_state: Updated temporal state for video (None for images)
                
        Raises:
            RuntimeError: If input dimensions are invalid
            NotImplementedError: If temporal mode is enabled but not implemented
            
        Example:
            >>> model = MultitaskPerceptionModel(cfg)
            >>> images = torch.randn(2, 3, 640, 640)
            >>> 
            >>> # Training
            >>> model.train()
            >>> targets = {'boxes': ..., 'labels': ..., 'seg_mask': ...}
            >>> outputs, losses, _ = model(images, targets)
            >>> 
            >>> # Inference
            >>> model.eval()
            >>> with torch.no_grad():
            >>>     outputs, _, _ = model(images)
            >>>     detections = outputs['detection']
            >>>     segmentation = outputs['segmentation']
        """
        # Extract features from backbone
        if self.use_temporal and images.dim() == 5:
            # Video mode: process temporal sequence (future feature)
            features, new_state = self._forward_video(images, temporal_state)
        else:
            # Image mode: standard forward pass
            if images.dim() != 4:
                raise RuntimeError(
                    f"Expected 4D image tensor [B, C, H, W], got shape {images.shape}"
                )
            features = self.backbone(images)
            new_state = None
        
        # Apply neck if present (currently disabled)
        if self.neck is not None:
            features = self.neck(features)
        
        outputs: dict[str, torch.Tensor] = {}
        losses: dict[str, torch.Tensor] = {}
        
        # Detection head
        if "detection" in self.enabled_tasks:
            if self.training and targets is not None:
                det_out, det_loss = self.heads["detection"](features, targets)
                losses.update(det_loss)
            else:
                det_out = self.heads["detection"](features)
            outputs["detection"] = det_out
        
        # Segmentation head
        if "segmentation" in self.enabled_tasks:
            seg_targets = targets.get("seg_mask") if targets else None
            if self.training and seg_targets is not None:
                seg_out, seg_loss = self.heads["segmentation"](features, seg_targets)
                losses["seg_loss"] = seg_loss
            else:
                seg_out = self.heads["segmentation"](features)
            outputs["segmentation"] = seg_out
        
        # Depth head
        if "depth" in self.enabled_tasks:
            depth_targets = targets.get("depth") if targets else None
            if self.training and depth_targets is not None:
                depth_out, depth_loss = self.heads["depth"](features, depth_targets)
                losses["depth_loss"] = depth_loss
            else:
                depth_out = self.heads["depth"](features)
            outputs["depth"] = depth_out
        
        return outputs, losses, new_state
    
    def _forward_video(
        self,
        video_frames: torch.Tensor,
        temporal_state: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Process video sequence with temporal modeling.
        
        This method will be implemented in the future (2-3 months) to support
        video-based inference with temporal consistency.
        
        Args:
            video_frames: Video tensor [B, T, C, H, W]
            temporal_state: Previous temporal state
            
        Returns:
            features: Temporally aggregated features
            new_state: Updated temporal state
            
        Raises:
            NotImplementedError: Always raised (feature not yet implemented)
            
        Note:
            To implement video support:
            1. Add temporal module (LSTM/Transformer/3DConv)
            2. Process frame features through temporal module
            3. Return aggregated features and updated state
            
        Example (future implementation):
            >>> # B, T, C, H, W = video_frames.shape
            >>> # frame_features = [self.backbone(video_frames[:, t]) for t in range(T)]
            >>> # temporal_features, new_state = self.temporal(frame_features, temporal_state)
            >>> # return temporal_features, new_state
        """
        raise NotImplementedError(
            "Video processing not yet implemented. "
            "This feature is planned for 2-3 months. "
            "For now, process video frame-by-frame or disable temporal mode."
        )
