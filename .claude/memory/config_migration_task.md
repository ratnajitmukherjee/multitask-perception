---
name: Config Schema Migration — COMPLETED
description: Config schema migration from old to new is DONE. Option A (forward) applied 2026-04-14. This memory kept for historical reference.
type: project
---

**Status: COMPLETED on 2026-04-14** (commit `97fc8cf`)

**What was done:** Option A (migrate forward) — 58 files changed, 644 insertions, 960 deletions.

**Key mappings applied:**
| Old | New |
|-----|-----|
| `TASK.TYPE = "Detection"` | `TASK.ENABLED = ["detection"]` |
| `MODEL.HEAD.DET_NAME = "CenterNetHead"` | `MODEL.HEADS.DETECTION.NAME = "CenterNet"` |
| `MODEL.NUM_CLASSES` | `MODEL.HEADS.DETECTION.NUM_CLASSES` |
| `MODEL.NUM_SEG_CLASSES` | `MODEL.HEADS.SEGMENTATION.NUM_CLASSES` |
| `DATA_LOADER.*` | `DATALOADER.*` |
| `SOLVER.LR` | `SOLVER.BASE_LR` |
| `SOLVER.NAME` | `SOLVER.OPTIMIZER` |
| `SCHEDULER.TYPE` | `SOLVER.LR_SCHEDULER` |
| `SCHEDULER.MIN_LR` | `SOLVER.MIN_LR` |
| `MODEL.BACKBONE.OUT_CHANNEL` | `MODEL.BACKBONE.OUT_CHANNELS` (tuple) |
| `MODEL.PAN.*` | `MODEL.HEADS.DETECTION.PAN.*` |
| `MODEL.LOSS.*` | `MODEL.HEADS.DETECTION.LOSS.*` |

**Design decisions:**
- `MODEL.HEADS.DETECTION` and `MODEL.HEADS.SEGMENTATION` use `new_allowed=True` for head-specific keys from sub-configs/YAMLs
- `MODEL.BACKBONE` uses `new_allowed=True` for architecture-specific keys
- CenterNet `OUT_CHANNELS` stored as tuple `(1024,)`, code takes `[-1]` element
- sub_cfg_dict keys changed from `"CenterNetHead"/"NanoDetHead"` to `"CenterNet"/"NanoDet"`
- `_get_task_type(cfg)` helper derives legacy "Detection"/"Segmentation"/"Multitask" string from TASK.ENABLED list

**SSD removal (same commit):** 3 YAML configs, box_utils.py (10 functions, zero external consumers), SSDTargetTransform, transform_SSD, MODEL.PRIORS, MODEL.BACKBONE.EXTRA — all dead code.

**How to apply:** This is historical reference. The migration is done and verified. If adding new heads or config keys, follow the NEW schema patterns.