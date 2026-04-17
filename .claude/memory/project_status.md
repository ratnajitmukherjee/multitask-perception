---
name: Project Status and Phase Tracking
description: Current development phase, what is complete, what is in progress, and what is next — updated 2026-04-17
type: project
---

**Last updated:** 2026-04-17

**Phase 1: Infrastructure Setup — COMPLETE**
- Project structure, core model architecture (MultitaskPerceptionModel)
- 3 backbones implemented (HarDNet, VoVNet, MobileNetV3)
- Configuration system (YACS), Docker support, Poetry environment
- Placeholder heads created

**Phase 2: Migration from Old Codebase — COMPLETE**
- Detection heads (NanoDet, CenterNet) ported and config-migrated
- Segmentation heads (DeepLabV3, ESPNetV2, SegFormer) ported (placeholders)
- Solver, losses, layers modules ported
- **Config schema migration DONE (Option A — forward)**: 58 files changed, 32 YAML configs verified
- **SSD dead code removed**: 3 YAML configs, box_utils.py, SSDTargetTransform, transform_SSD, MODEL.PRIORS
- Commit: `97fc8cf` on 2026-04-14
- GitHub Wiki published (7 pages), CLAUDE.md added, Obsidian vault updated

**Why:** Config schema was the #1 blocker. Now unified on NEW schema (TASK.ENABLED, MODEL.HEADS.*, DATALOADER.*, SOLVER.OPTIMIZER/BASE_LR/LR_SCHEDULER). No old schema references remain.

**How to apply:** Migration is done. Next work should focus on Phase 2.5 prerequisites then Phase 3.

**Phase 2.5: Pre-Training Prerequisites — NEXT**
1. Pretrained backbone weights (HIGH) — currently random init only
2. Device-agnostic fixes (MEDIUM) — replace hard .cuda() calls
3. Smoke test — forward pass a dummy batch end-to-end

**Remaining known issues:**
1. Pretrained backbone weights not implemented (random init only)
2. Hard `.cuda()` calls in CenterNet/NanoDet losses (not device-agnostic)
3. Undefined `CenternetHelper` ONNX references in CenterNet decode
4. Segmentation heads are placeholder stubs
5. Depth head not implemented

**Pending phases:**
- Phase 3: Dataset preparation (KITTI, nuScenes, BDD100K loaders + pseudo-labeling)
- Phase 4: Training (detection-only → seg-only → multitask)
- Phase 5: Depth integration
- Phase 6: Optimization (hyperparams, compression, ONNX export)
- Phase 7: Video support (temporal module, tracking — 2-3 months out)

**Pre-release cleanup task:** Remove `.claude/memory/` from repo before going public.