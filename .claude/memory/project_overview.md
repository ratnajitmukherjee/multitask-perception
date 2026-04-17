---
name: Multitask Perception Project Overview
description: Unified multitask perception system for autonomous driving — detection, segmentation, depth estimation with shared backbone
type: project
---

**Goal:** Build a unified multitask perception system for autonomous driving that performs:
1. Object Detection (vehicles, pedestrians, traffic signs)
2. Semantic Segmentation (drivable area, lanes, sidewalks)
3. Monocular Depth Estimation (distance to objects/surfaces)

**Why:** Single shared backbone for efficiency on resource-constrained hardware (GTX 1080 Ti). Video-ready architecture planned for 2-3 months out.

**How to apply:** All work should consider VRAM constraints (11GB), multitask loss balancing, and future depth/video extensibility.

**Key architecture:**
```
Input Image → Backbone (shared) → [Optional Neck/FPN] → [Optional Temporal] →
  ├── Detection Head (NanoDet or CenterNet)
  ├── Segmentation Head (SegFormer, DeepLabV3, or ESPNetV2)
  └── Depth Head (not yet implemented)
```

**Tech stack:** Python 3.12, PyTorch 2.2.0, CUDA 11.8, Poetry 2.3.2, timm, albumentations, yacs (config), wandb + tensorboard

**Backbones kept:** HarDNet-68/85, VoVNet-27/39, MobileNetV3-S/L
**Detection heads:** NanoDet, CenterNet
**Segmentation heads:** SegFormer-B0, DeepLabV3, ESPNetV2
**Depth head:** SimpleDecoder (to be implemented)

**Dataset strategy:** Unified multitask dataset from KITTI + nuScenes + BDD100K + Cityscapes + Mapillary Vistas (~130K images). Pseudo-labeling for missing depth annotations using ZoeDepth-NK.

**Budget:** ~$185 total ($60 Claude Pro + $125 Lambda Labs cloud)