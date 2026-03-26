import torch
import torch.nn as nn

from multitask_perception.modeling import registry
from multitask_perception.modeling.heads.detection.nanodet.nanodet_loss import NanoDetLoss
from multitask_perception.modeling.heads.detection.nanodet.nanodet_postprocess import (
    NanoDetPostProcess,
)
from multitask_perception.modeling.heads.detection.nanodet.nanodet_predictor import (
    NanoDetPredictor,
)
from multitask_perception.modeling.heads.detection.nanodet.pan import PAN


@registry.HEADS.register("NanoDetHead")
class NanoDetHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.predictor = NanoDetPredictor(cfg)
        self.loss_evaluator = NanoDetLoss(cfg)
        self.post_processor = NanoDetPostProcess(cfg)
        self.pan = PAN(cfg)

    def forward(self, features, targets=None):
        features = self.pan(features)
        output = self.predictor(features)
        if torch.onnx.is_in_onnx_export():
            return output, {}
        if self.training:
            return self._forward_train(output, targets)
        else:
            return self._forward_test(output)

    def _forward_train(self, output, targets):
        gt_boxes, gt_labels = targets["boxes"], targets["labels"]
        loss, loss_dict = self.loss_evaluator(output, gt_labels, gt_boxes)
        return output, loss_dict

    def _forward_test(self, output):
        detections = self.post_processor(output)
        return detections, {}
