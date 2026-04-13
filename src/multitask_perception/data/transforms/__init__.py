from multitask_perception.data.transforms.target_transform import (
    CenterNetHeadTransform,
)
from multitask_perception.data.transforms.target_transform import (
    NanoDetTargetTransform,
)
from multitask_perception.data.transforms.transforms import *


"""
First we list out the transforms for each head
1) CenterNet
2) NanoDet
"""


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
            RandomAffine(task_type=_get_task_type(cfg)),
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
            RandomAffine(task_type=_get_task_type(cfg)),
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


def _get_task_type(cfg):
    """Derive legacy task type string from TASK.ENABLED list.

    RandomAffine and other transforms expect a string: "Detection",
    "Segmentation", or "Multitask".
    """
    enabled = cfg.TASK.ENABLED
    has_det = "detection" in enabled
    has_seg = "segmentation" in enabled
    if has_det and has_seg:
        return "Multitask"
    elif has_det:
        return "Detection"
    elif has_seg:
        return "Segmentation"
    return "Detection"


def build_transforms(cfg, is_train=True):
    """
    Image transforms (for all heads)
    :param cfg: config file
    :param is_train: train or test mode
    :return: return the transformations for the calling head
    """
    if "detection" in cfg.TASK.ENABLED:
        det_name = cfg.MODEL.HEADS.DETECTION.NAME.lower()
        if "centernet" in det_name:
            return transform_CenterNet(cfg, is_train)
        elif "nanodet" in det_name:
            return transform_NanoDet(cfg, is_train)
        else:
            raise NotImplementedError(
                "Transformation for detection head {} not implemented.".format(
                    cfg.MODEL.HEADS.DETECTION.NAME
                )
            )
    elif "segmentation" in cfg.TASK.ENABLED:
        return transform_Segmentation(cfg, is_train)
    else:
        raise NotImplementedError("Incorrect Task in configuration")


def build_target_transform(cfg):
    """
    Target transforms (for all heads) - ground truth boxes and labels or heatmaps
    :param cfg: config file
    :return: return the transformations for the calling head
    """
    if "detection" in cfg.TASK.ENABLED:
        det_name = cfg.MODEL.HEADS.DETECTION.NAME.lower()
        if "centernet" in det_name:
            return CenterNetHeadTransform(cfg)
        elif "nanodet" in det_name:
            return NanoDetTargetTransform()
        else:
            raise NotImplementedError(
                "Target transformation for detection head {} not implemented.".format(
                    cfg.MODEL.HEADS.DETECTION.NAME
                )
            )
