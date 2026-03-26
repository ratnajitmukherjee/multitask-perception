from multitask_perception.data.transforms.target_transform import (
    CenterNetHeadTransform,
)
from multitask_perception.data.transforms.target_transform import (
    NanoDetTargetTransform,
)
from multitask_perception.data.transforms.target_transform import (
    SSDTargetTransform,
)
from multitask_perception.data.transforms.transforms import *
# TODO: prior_box not ported — from multitask_perception.modeling.anchors.prior_box import PriorBox


"""
First we list out the transforms for each head
1) CenterNet
2) NanoDet
3) SSDBoxHead
"""


def transform_SSD(cfg, is_train=True):
    """
    Data transforms for SSD box head
    :param cfg: config file
    :param is_train: parameter to differentiate between train and test mode
    :return: transformed data
    """
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(cfg.INPUT.PIXEL_MEAN),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    transform = Compose(transform)
    return transform


def transform_NanoDet(cfg, is_train=True):
    """
    Data transforms for Nanodet box head
    :param cfg: config file
    :param is_train: parameter to differentiate between train and test mode
    :return: transformed data
    """
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            RandomSampleCrop(),
            RandomMirror(),
            RandomAffine(task_type=cfg.TASK.TYPE),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            Remake(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    else:
        transform = [
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            Remake(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    transform = Compose(transform)
    return transform


def transform_CenterNet(cfg, is_train=True):
    """
    Data transforms for CenterNetHead
    :param cfg: config file
    :param is_train: parameter to differentiate between train and test mode
    :return: transformed data
    """
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    transform = Compose(transform)
    return transform


"""
We have a single transformation function for all segmentation heads.
NOTE: if required we can make separate transformation for each segmentation head.
"""


def transform_Segmentation(cfg, is_train=True):
    """
    Data transforms for Segmentation head
    :param cfg: config file
    :param is_train: parameter to differentiate between train and test mode
    :return: transformed data
    """
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            # TODO: Make RandomSampleCrop work with segmentation
            # RandomSampleCrop(),
            RandomMirror(),
            RandomAffine(task_type=cfg.TASK.TYPE),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            Remake(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    else:
        transform = [
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            Remake(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    transform = Compose(transform)
    return transform


"""
Now, the main controller functions to call Head-Specific transforms
"""


def build_transforms(cfg, is_train=True):
    """
    Image transforms (for all heads)
    :param cfg: config file
    :param is_train: train or test mode
    :return: return the transformations for the calling head
    """
    if cfg.TASK.TYPE == "Detection" or cfg.TASK.TYPE == "Multitask":
        if "SSD" in cfg.MODEL.HEAD.DET_NAME:
            return transform_SSD(cfg, is_train)
        elif "CenterNetHead" in cfg.MODEL.HEAD.DET_NAME:
            return transform_CenterNet(cfg, is_train)
        elif "NanoDetHead" in cfg.MODEL.HEAD.DET_NAME:
            return transform_NanoDet(cfg, is_train)
        else:
            raise NotImplementedError(
                "Transformation for detection head {} not implemented.".format(
                    cfg.MODEL.HEAD.DET_NAME
                )
            )
    elif cfg.TASK.TYPE == "Segmentation":
        return transform_Segmentation(cfg, is_train)
    else:
        raise NotImplementedError("Incorrect Task in configuration")


def build_target_transform(cfg):
    """
    Target transforms (for all heads) - ground truth boxes and labels or heatmaps
    :param cfg: config file
    :return: return the transformations for the calling head
    """
    if cfg.TASK.TYPE == "Detection" or cfg.TASK.TYPE == "Multitask":
        if "SSD" in cfg.MODEL.HEAD.DET_NAME:
            return SSDTargetTransform(
                PriorBox(cfg)(),
                cfg.MODEL.CENTER_VARIANCE,
                cfg.MODEL.SIZE_VARIANCE,
                cfg.MODEL.THRESHOLD,
            )
        if "CenterNetHead" in cfg.MODEL.HEAD.DET_NAME:
            return CenterNetHeadTransform(cfg)
        elif "NanoDetHead" in cfg.MODEL.HEAD.DET_NAME:
            return NanoDetTargetTransform()
        else:
            raise NotImplementedError(
                "Target transformation for detection head {} not implemented.".format(
                    cfg.MODEL.HEAD.DET_NAME
                )
            )
