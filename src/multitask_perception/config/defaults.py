"""
Default configuration for multitask perception.

This module defines all configuration parameters with their default values
using YACS (Yet Another Configuration System).
"""

from yacs.config import CfgNode as CN


_C = CN()

# ---------------------------------------------------------------------------- #
# Task Configuration
# ---------------------------------------------------------------------------- #
_C.TASK = CN()
_C.TASK.ENABLED = ["detection", "segmentation"]  # List of enabled tasks

# ---------------------------------------------------------------------------- #
# Model Configuration
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "MultitaskPerceptionModel"

# Backbone
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "hardnet68"  # Options: hardnet68, hardnet85, vovnet27_slim, vovnet39, mobilenet_v3_small, mobilenet_v3_large
_C.MODEL.BACKBONE.PRETRAINED = False
_C.MODEL.BACKBONE.FREEZE = False

# Neck (optional, currently disabled)
_C.MODEL.NECK = CN()
_C.MODEL.NECK.ENABLED = False
_C.MODEL.NECK.NAME = "FPN"

# Temporal module (for video, future feature)
_C.MODEL.TEMPORAL = CN()
_C.MODEL.TEMPORAL.ENABLED = False
_C.MODEL.TEMPORAL.TYPE = "lstm"  # Options: lstm, gru, transformer
_C.MODEL.TEMPORAL.HIDDEN_DIM = 256
_C.MODEL.TEMPORAL.NUM_LAYERS = 2

# Task Heads
_C.MODEL.HEADS = CN()

# Detection Head
_C.MODEL.HEADS.DETECTION = CN()
_C.MODEL.HEADS.DETECTION.NAME = "NanoDet"  # Options: NanoDet, CenterNet
_C.MODEL.HEADS.DETECTION.NUM_CLASSES = 10
_C.MODEL.HEADS.DETECTION.USE_DCN = False  # Use Deformable Conv (requires compiled C++/CUDA ext); False uses standard Conv2d

# Segmentation Head
_C.MODEL.HEADS.SEGMENTATION = CN()
_C.MODEL.HEADS.SEGMENTATION.NAME = "SegFormer"  # Options: SegFormer, DeepLabV3, ESPNetV2
_C.MODEL.HEADS.SEGMENTATION.NUM_CLASSES = 19

# Depth Head
_C.MODEL.HEADS.DEPTH = CN()
_C.MODEL.HEADS.DEPTH.NAME = "SimpleDecoder"
_C.MODEL.HEADS.DEPTH.OUTPUT_CHANNELS = 1

# Loss Weights
_C.MODEL.LOSS_WEIGHTS = CN()
_C.MODEL.LOSS_WEIGHTS.DETECTION = 1.0
_C.MODEL.LOSS_WEIGHTS.SEGMENTATION = 2.0
_C.MODEL.LOSS_WEIGHTS.DEPTH = 1.5

# ---------------------------------------------------------------------------- #
# Input Configuration
# ---------------------------------------------------------------------------- #
_C.INPUT = CN()
_C.INPUT.IMAGE_SIZE = 640  # Input image size
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]  # ImageNet std
_C.INPUT.FORMAT = "RGB"  # or "BGR"

# ---------------------------------------------------------------------------- #
# Dataset Configuration
# ---------------------------------------------------------------------------- #
_C.DATASETS = CN()
_C.DATASETS.TRAIN = ["kitti_train"]  # List of training datasets
_C.DATASETS.VAL = ["kitti_val"]  # List of validation datasets
_C.DATASETS.TEST = ["kitti_test"]  # List of test datasets

# ---------------------------------------------------------------------------- #
# DataLoader Configuration
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.BATCH_SIZE = 8
_C.DATALOADER.SHUFFLE = True
_C.DATALOADER.PIN_MEMORY = True
_C.DATALOADER.DROP_LAST = True

# ---------------------------------------------------------------------------- #
# Solver (Optimizer & Scheduler) Configuration
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = "AdamW"  # Options: SGD, Adam, AdamW
_C.SOLVER.BASE_LR = 1e-3
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.MOMENTUM = 0.9  # For SGD
_C.SOLVER.BETAS = (0.9, 0.999)  # For Adam/AdamW

# Learning Rate Scheduler
_C.SOLVER.LR_SCHEDULER = "CosineAnnealing"  # Options: CosineAnnealing, MultiStep, OneCycle
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_FACTOR = 0.001
_C.SOLVER.MAX_ITER = 100000

# MultiStep scheduler
_C.SOLVER.LR_STEPS = [60000, 80000]
_C.SOLVER.GAMMA = 0.1

# Gradient clipping
_C.SOLVER.CLIP_GRAD = CN()
_C.SOLVER.CLIP_GRAD.ENABLED = False
_C.SOLVER.CLIP_GRAD.MAX_NORM = 1.0

# ---------------------------------------------------------------------------- #
# Training Configuration
# ---------------------------------------------------------------------------- #
_C.TRAINING = CN()
_C.TRAINING.CHECKPOINT_PERIOD = 5000  # Save checkpoint every N iterations
_C.TRAINING.EVAL_PERIOD = 5000  # Evaluate every N iterations
_C.TRAINING.LOG_PERIOD = 50  # Log every N iterations
_C.TRAINING.MIXED_PRECISION = True  # Use FP16 training
_C.TRAINING.RESUME = ""  # Path to checkpoint to resume from

# ---------------------------------------------------------------------------- #
# Testing Configuration
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1
_C.TEST.EVAL_METRICS = ["mAP", "mIoU"]  # Metrics to compute

# ---------------------------------------------------------------------------- #
# Output Configuration
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "./outputs"
_C.LOGGER = CN()
_C.LOGGER.NAME = "multitask_perception"
_C.LOGGER.DEBUG_MODE = False

# ---------------------------------------------------------------------------- #
# Misc
# ---------------------------------------------------------------------------- #
_C.SEED = 42
_C.CUDNN_BENCHMARK = True
_C.CUDNN_DETERMINISTIC = False


def get_cfg_defaults() -> CN:
    """
    Get a copy of the default configuration.
    
    Returns:
        A yacs CfgNode object containing default configuration.
        
    Example:
        >>> cfg = get_cfg_defaults()
        >>> cfg.MODEL.BACKBONE.NAME = 'vovnet39'
        >>> cfg.merge_from_file('configs/experiment.yaml')
        >>> cfg.freeze()
    """
    return _C.clone()
