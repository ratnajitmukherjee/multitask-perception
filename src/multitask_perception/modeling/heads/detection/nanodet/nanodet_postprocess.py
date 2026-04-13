import numpy as np
import torch
import torch.nn as nn

from multitask_perception.modeling.heads.detection.nanodet.utils.anchor_generator import (
    AnchorGenerator,
)
from multitask_perception.modeling.heads.detection.nanodet.utils.box_transform import (
    distance2bbox,
)
from multitask_perception.modeling.heads.detection.nanodet.utils.helper_func import (
    anchor_center,
)
from multitask_perception.modeling.heads.detection.nanodet.utils.integral import Integral
from multitask_perception.modeling.heads.detection.nanodet.utils.nms import multiclass_nms
from multitask_perception.structures.container import Container


class NanoDetPostProcess(nn.Module):
    def __init__(
        self,
        cfg,
        anchor_ratios=[1.0],
        anchor_base_sizes=None,
    ):
        super().__init__()
        self.img_size = (cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE)
        self.num_classes = cfg.MODEL.HEADS.DETECTION.NUM_CLASSES
        self.cls_out_channels = cfg.MODEL.HEADS.DETECTION.NUM_CLASSES
        self.anchor_strides = cfg.MODEL.HEADS.DETECTION.STRIDES
        self.reg_max = cfg.MODEL.HEADS.DETECTION.REG_MAX

        self.distribution_project = Integral(self.reg_max)

        octave_base_scale = cfg.MODEL.HEADS.DETECTION.LOSS.OCTAVE_BASE_SCALE
        scales_per_octave = cfg.MODEL.HEADS.DETECTION.LOSS.SCALES_PER_OCTAVE

        self.anchor_base_sizes = (
            list(cfg.MODEL.HEADS.DETECTION.STRIDES)
            if anchor_base_sizes is None
            else anchor_base_sizes
        )
        self.anchor_generators = []
        octave_scales = np.array(
            [2 ** (i / scales_per_octave) for i in range(scales_per_octave)]
        )
        anchor_scales = octave_scales * octave_base_scale
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, anchor_scales, anchor_ratios)
            )

    def forward(self, preds):
        cls_scores, bbox_preds = preds  # here
        result_list = self.get_bboxes(cls_scores, bbox_preds)

        return result_list

    def get_bboxes(self, cls_scores, bbox_preds, rescale=False):
        assert len(cls_scores) == len(bbox_preds)  # here
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(
                cls_scores[i].size()[-2:], self.anchor_strides[i], device=device
            )
            for i in range(num_levels)
        ]

        input_height, input_width = self.img_size

        input_shape = [input_height, input_width]

        result_list = []
        for img_id in range(cls_scores[0].shape[0]):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            scale_factor = 1
            dets = self.get_bboxes_single(
                cls_score_list,
                bbox_pred_list,
                mlvl_anchors,
                input_shape,
                scale_factor,
                rescale,
            )

            result_list.append(dets)
        return result_list

    def get_bboxes_single(
        self,
        cls_scores,
        bbox_preds,
        mlvl_anchors,
        img_shape,  # input shape!!!!
        scale_factor,
        rescale=False,
    ):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for stride, cls_score, bbox_pred, anchors in zip(
            self.anchor_strides, cls_scores, bbox_preds, mlvl_anchors
        ):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = (
                cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            )
            bbox_pred = bbox_pred.permute(1, 2, 0)
            bbox_pred = self.distribution_project(bbox_pred) * stride

            # nms_pre = cfg.get('nms_pre', -1)
            nms_pre = 1000
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            bboxes = distance2bbox(
                anchor_center(anchors), bbox_pred, max_shape=img_shape
            )
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        mlvl_scores = torch.cat(mlvl_scores)
        # add a dummy background class at the end of all labels, same with mmdetection2.0
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        det_bboxes, det_labels, det_scores = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            score_thr=0.05,
            nms_cfg=dict(type="nms", iou_threshold=0.6),
            max_num=100,
        )
        img_height, img_width = self.img_size
        container = Container(boxes=det_bboxes, labels=det_labels, scores=det_scores)
        container.img_width = img_width
        container.img_height = img_height
        return container
