import torch
import torch.nn as nn

from multitask_perception.modeling.heads.detection.nanodet.utils.conv import (
    DepthwiseConvModule,
)
from multitask_perception.modeling.heads.detection.nanodet.utils.helper_func import (
    multi_apply,
)
from multitask_perception.modeling.heads.detection.nanodet.utils.init_weights import (
    normal_init,
)


class NanoDetPredictor(nn.Module):
    """
    Modified from GFL, use same loss functions but much lightweight convolution heads
    """

    def __init__(
        self,
        cfg,
        activation="LeakyReLU",
    ):
        super().__init__()
        self.share_cls_reg = cfg.MODEL.HEADS.DETECTION.SHARE_CLS_REG
        self.activation = activation
        self.stacked_convs = cfg.MODEL.HEADS.DETECTION.STACKED_CONVS
        self.in_channels = cfg.MODEL.HEADS.DETECTION.PAN.OUT_CHANNELS
        self.anchor_strides = cfg.MODEL.HEADS.DETECTION.STRIDES
        self.feat_channels = cfg.MODEL.HEADS.DETECTION.FEAT_CHANNELS
        self.cls_out_channels = cfg.MODEL.HEADS.DETECTION.NUM_CLASSES
        self.num_classes = cfg.MODEL.HEADS.DETECTION.NUM_CLASSES
        self.reg_max = cfg.MODEL.HEADS.DETECTION.REG_MAX
        self.norm_cfg = dict(type=cfg.MODEL.HEADS.DETECTION.NORM_CFG_TYPE)
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for _ in self.anchor_strides:
            cls_convs, reg_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

        self.gfl_cls = nn.ModuleList(
            [
                nn.Conv2d(
                    self.feat_channels,
                    self.cls_out_channels + 4 * (self.reg_max + 1)
                    if self.share_cls_reg
                    else self.cls_out_channels,
                    1,
                    padding=0,
                )
                for _ in self.anchor_strides
            ]
        )

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                DepthwiseConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                )
            )
            if not self.share_cls_reg:
                reg_convs.append(
                    DepthwiseConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None,
                        activation=self.activation,
                    )
                )

        return cls_convs, reg_convs

    def init_weights(self):
        for seq in self.cls_convs:
            for m in seq:
                normal_init(m.depthwise, std=0.01)
                normal_init(m.pointwise, std=0.01)
        for seq in self.reg_convs:
            for m in seq:
                normal_init(m.depthwise, std=0.01)
                normal_init(m.pointwise, std=0.01)
        bias_cls = -4.595  # 用0.01的置信度初始化
        for i in range(len(self.anchor_strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
            # normal_init(self.gfl_reg[i], std=0.01)
        print("Finish initialize Lite GFL Head.")

    def forward(self, feats):
        return multi_apply(
            self.forward_single,
            feats,
            self.cls_convs,
            self.reg_convs,
            self.gfl_cls,
        )

    def forward_single(self, x, cls_convs, reg_convs, gfl_cls, gfl_reg=None):
        cls_feat = x
        reg_feat = x
        for cls_conv in cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in reg_convs:
            reg_feat = reg_conv(reg_feat)
        if self.share_cls_reg:
            feat = gfl_cls(cls_feat)
            cls_score, bbox_pred = torch.split(
                feat, [self.cls_out_channels, 4 * (self.reg_max + 1)], dim=1
            )
        else:
            cls_score = gfl_cls(cls_feat)
            bbox_pred = gfl_reg(reg_feat)

        if torch.onnx.is_in_onnx_export():
            cls_score = (
                torch.sigmoid(cls_score)
                .reshape(1, self.num_classes, -1)
                .permute(0, 2, 1)
            )
            bbox_pred = bbox_pred.reshape(1, (self.reg_max + 1) * 4, -1).permute(
                0, 2, 1
            )
        return cls_score, bbox_pred
