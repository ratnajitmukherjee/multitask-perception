"""
Head-specific configuration nodes.

Each detection head may introduce its own config keys (e.g. NanoDet needs
MODEL.PAN, MODEL.HEAD.REG_MAX, etc.). These sub-configs use new_allowed=True
so that YAML experiment files can add arbitrary keys without modifying defaults.

The sub_cfg_dict maps head names (as they appear in the YAML config under
MODEL.HEAD.DET_NAME) to their respective CfgNode.
"""
from yacs.config import CfgNode as CN


# CenterNet: empty node, all keys come from the YAML config
centernet_cfg = CN(new_allowed=True)

# NanoDet: provides defaults for PAN, HEAD, and LOSS sub-configs
nanodet_cfg = CN(new_allowed=True)
nanodet_cfg.MODEL = CN(new_allowed=True)
nanodet_cfg.MODEL.PAN = CN(new_allowed=True)
nanodet_cfg.MODEL.PAN.OUT_CHANNELS = 96
nanodet_cfg.MODEL.HEAD = CN(new_allowed=True)
nanodet_cfg.MODEL.HEAD.FEAT_CHANNELS = 96
nanodet_cfg.MODEL.HEAD.STACKED_CONVS = 2
nanodet_cfg.MODEL.HEAD.SHARE_CLS_REG = True
nanodet_cfg.MODEL.HEAD.REG_MAX = 7
nanodet_cfg.MODEL.HEAD.STRIDES = [8, 16, 32]
nanodet_cfg.MODEL.HEAD.NORM_CFG_TYPE = "BN"
nanodet_cfg.MODEL.LOSS = CN(new_allowed=True)
nanodet_cfg.MODEL.LOSS.OCTAVE_BASE_SCALE = 5
nanodet_cfg.MODEL.LOSS.SCALES_PER_OCTAVE = 1


# Mapping from detection head name to its sub-config
sub_cfg_dict = {
    "CenterNetHead": centernet_cfg,
    "NanoDetHead": nanodet_cfg,
}
