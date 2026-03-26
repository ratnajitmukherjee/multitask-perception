import numpy as np
import torch

from multitask_perception.modeling.heads.detection.centernet.utils import (
    draw_umich_gaussian,
)
from multitask_perception.modeling.heads.detection.centernet.utils import gaussian2D
from multitask_perception.modeling.heads.detection.centernet.utils import gaussian_radius
from multitask_perception.utils import box_utils


class SSDTargetTransform:
    def __init__(
        self, center_form_priors, center_variance, size_variance, iou_threshold
    ):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(
            center_form_priors
        )
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(
            gt_boxes, gt_labels, self.corner_form_priors, self.iou_threshold
        )
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(
            boxes, self.center_form_priors, self.center_variance, self.size_variance
        )
        return locations, labels


class NanoDetTargetTransform:
    def __init__(
        self,
    ):
        pass

    def __call__(self, boxes, classes):
        return np.array(boxes), np.array(classes)


class CenterNetHeadTransform:
    def __init__(self, cfg):
        self.max_objs = 128
        self.num_classes = cfg.MODEL.NUM_CLASSES

        self.output_h = (
            np.power(2, cfg.MODEL.HEAD.NUM_DECONV_LAYERS)
            * cfg.MODEL.HEAD.BACKBONE_FEATURE
        )
        self.output_w = (
            np.power(2, cfg.MODEL.HEAD.NUM_DECONV_LAYERS)
            * cfg.MODEL.HEAD.BACKBONE_FEATURE
        )

    def __call__(self, gt_boxes, gt_labels):
        num_objs = min(len(gt_labels), self.max_objs)
        self.wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        self.reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        self.ind = np.zeros(self.max_objs, dtype=np.int64)
        self.reg_mask = np.zeros(self.max_objs, dtype=np.uint8)
        self.cat_spec_wh = np.zeros(
            (self.max_objs, self.num_classes * 2), dtype=np.float32
        )
        self.cat_spec_mask = np.zeros(
            (self.max_objs, self.num_classes * 2), dtype=np.uint8
        )
        self.draw_gaussian = draw_umich_gaussian
        self.hm = np.zeros(
            (self.num_classes, self.output_h, self.output_w), dtype=np.float32
        )

        for k in range(num_objs):
            bbox = gt_boxes[k]
            cls_id = int(gt_labels[k])
            bbox = bbox * self.output_h
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((np.math.ceil(h), np.math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32
                )
                ct_int = ct.astype(np.int32)
                out = self.draw_gaussian(self.hm[cls_id], ct_int, radius)
                self.wh[k] = 1.0 * w, 1.0 * h
                self.ind[k] = ct_int[1] * self.output_w + ct_int[0]
                self.reg[k] = ct - ct_int
                self.reg_mask[k] = 1

        ret = {
            "hm": self.hm,
            "reg_mask": self.reg_mask,
            "ind": self.ind,
            "wh": self.wh,
            "reg": self.reg,
        }
        return ret, list()
