"""
Head-specific configuration nodes.

Each detection head may introduce its own config keys (e.g. NanoDet needs
PAN, REG_MAX, STRIDES, etc.). These sub-configs use new_allowed=True
so that YAML experiment files can add arbitrary keys without modifying defaults.

The sub_cfg_dict maps head names (as they appear in the YAML config under
MODEL.HEADS.DETECTION.NAME) to their respective CfgNode.
"""
from yacs.config import CfgNode as CN


# CenterNet: empty node, all keys come from the YAML config
centernet_cfg = CN(new_allowed=True)

# NanoDet: provides defaults for detection head and loss sub-configs
nanodet_cfg = CN(new_allowed=True)
nanodet_cfg.MODEL = CN(new_allowed=True)
nanodet_cfg.MODEL.HEADS = CN(new_allowed=True)
nanodet_cfg.MODEL.HEADS.DETECTION = CN(new_allowed=True)
nanodet_cfg.MODEL.HEADS.DETECTION.FEAT_CHANNELS = 96
nanodet_cfg.MODEL.HEADS.DETECTION.STACKED_CONVS = 2
nanodet_cfg.MODEL.HEADS.DETECTION.SHARE_CLS_REG = True
nanodet_cfg.MODEL.HEADS.DETECTION.REG_MAX = 7
nanodet_cfg.MODEL.HEADS.DETECTION.STRIDES = [8, 16, 32]
nanodet_cfg.MODEL.HEADS.DETECTION.NORM_CFG_TYPE = "BN"
nanodet_cfg.MODEL.HEADS.DETECTION.PAN = CN(new_allowed=True)
nanodet_cfg.MODEL.HEADS.DETECTION.PAN.OUT_CHANNELS = 96
nanodet_cfg.MODEL.HEADS.DETECTION.LOSS = CN(new_allowed=True)
nanodet_cfg.MODEL.HEADS.DETECTION.LOSS.OCTAVE_BASE_SCALE = 5
nanodet_cfg.MODEL.HEADS.DETECTION.LOSS.SCALES_PER_OCTAVE = 1


# Mapping from detection head name to its sub-config.
# Keys must match MODEL.HEADS.DETECTION.NAME values in YAML configs.
sub_cfg_dict = {
    "CenterNet": centernet_cfg,
    "NanoDet": nanodet_cfg,
}
