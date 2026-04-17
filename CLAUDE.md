# Multitask Perception - Claude Code Context

## Project
Unified multitask perception system for autonomous driving: object detection + semantic segmentation + monocular depth estimation with a shared backbone. Built by Ratnajit (Engineering Manager at TomTom, 6yr DL experience).

## Current Status (2026-04-17)
Phase 1 (Infrastructure) complete. Phase 2 (Migration) complete — config schema unified (Option A), SSD dead code removed, 32 YAML configs verified, GitHub Wiki published. **Next: Phase 2.5** — pretrained backbone weights, device-agnostic fixes, smoke test. Then Phase 3 (dataset preparation).

## Config Schema (UNIFIED — NEW)
The codebase uses a single, consistent config schema:
- `TASK.ENABLED = ["detection", "segmentation"]` — list of enabled tasks
- `MODEL.HEADS.DETECTION.NAME = "NanoDet"` — namespaced per-task heads
- `MODEL.HEADS.DETECTION.NUM_CLASSES = 10` — per-head num classes
- `MODEL.HEADS.SEGMENTATION.NAME = "SegFormer"` — segmentation head
- `DATALOADER.*` — no underscore
- `SOLVER.OPTIMIZER = "AdamW"` — not SOLVER.NAME
- `SOLVER.BASE_LR = 1e-3` — not SOLVER.LR
- `SOLVER.LR_SCHEDULER = "CosineAnnealing"` — not SCHEDULER.TYPE
- Head-specific keys (CenterNet deconv, NanoDet strides/PAN) come from `head_configs.py` sub-configs and YAMLs via `new_allowed=True`

## Tech Stack
- Python 3.12.12, PyTorch 2.2.0, CUDA 11.8, Poetry 2.3.2
- Config: YACS (yacs.config.CfgNode)
- Hardware: GTX 1080 Ti (11GB VRAM), Lambda Labs RTX 4090 (cloud)

## Package Structure
```
src/multitask_perception/
├── config/          # YACS defaults + head sub-configs (head_configs.py)
├── modeling/
│   ├── model.py     # MultitaskPerceptionModel
│   ├── backbones/   # HarDNet68/85, VoVNet27/39, MobileNetV3 (pretrained NOT implemented)
│   ├── heads/
│   │   ├── detection/   # CenterNet + NanoDet
│   │   ├── segmentation/ # DeepLabV3, ESPNetV2, SegFormer (PLACEHOLDERS)
│   │   └── depth/       # NOT IMPLEMENTED
│   ├── losses/      # FocalLoss, SegmentationLoss
│   └── layers/      # ESPNetV2 utils, SeparableConv, EfficientPyrPool
├── engine/          # trainer.py, inference.py
├── data/            # build.py, transforms, samplers
├── solver/          # SGD, Adam, LR schedulers
├── structures/      # Container class (dict-like, NOT ONNX-compatible)
└── utils/           # checkpoint, metrics, distributed, NMS, registry
```

## Entry Points (root level)
- `train.py` — Training with distributed support
- `test.py` — Evaluation with FLOPs/speed benchmarking
- `infer.py` — Single image/directory inference

## Key Design Patterns
- Registry pattern for backbones, heads, optimizers, schedulers
- Config-driven: YAML + CLI overrides, head sub-configs merged at startup
- Container class wraps detection outputs (boxes, labels, scores)
- Heads return (output, loss_dict) in training, (detections, {}) in inference
- `_get_task_type(cfg)` helper derives legacy task string from TASK.ENABLED list

## Documentation
- **GitHub Wiki:** https://github.com/ratnajitmukherjee/multitask-perception/wiki (7 pages: Architecture, Config, Getting Started, Structure, Training, Roadmap)
- **Obsidian Vault:** `~/Documents/Obsidian Vault/Multitask-Perception/` (PROJECT_OVERVIEW.md, Config Schema Migration Plan.md)

## Known Issues
1. Pretrained backbone weights not implemented (random init only)
2. Hard `.cuda()` calls in CenterNet/NanoDet losses (not device-agnostic)
3. Undefined `CenternetHelper` ONNX references in CenterNet decode
4. Segmentation heads are placeholder stubs
5. Depth head not implemented

## Claude Memory (Cross-Machine Sync)
- **Location:** `.claude/memory/` — checked into git for cross-machine context
- Contains: user profile, working style feedback, project status, config migration history, doc references
- **REMOVE before making repo public** (pre-release cleanup task)

## Code Style
- Black + isort formatting
- Google-style docstrings
- Type hints (Python 3.12 style)
- snake_case files, PascalCase classes
- Absolute imports preferred: `from multitask_perception.config import ...`
