import numpy as np
import torch
import torch.nn as nn

from multitask_perception.modeling.heads.detection.nanodet.atss_assigner import (
    ATSSAssigner,
)
from multitask_perception.modeling.heads.detection.nanodet.losses.gfocal_loss import (
    DistributionFocalLoss,
)
from multitask_perception.modeling.heads.detection.nanodet.losses.gfocal_loss import (
    QualityFocalLoss,
)
from multitask_perception.modeling.heads.detection.nanodet.losses.iou_loss import GIoULoss
from multitask_perception.modeling.heads.detection.nanodet.losses.iou_loss import (
    bbox_overlaps,
)
from multitask_perception.modeling.heads.detection.nanodet.utils.anchor_generator import (
    AnchorGenerator,
)
from multitask_perception.modeling.heads.detection.nanodet.utils.box_transform import (
    bbox2distance,
)
from multitask_perception.modeling.heads.detection.nanodet.utils.box_transform import (
    distance2bbox,
)
from multitask_perception.modeling.heads.detection.nanodet.utils.helper_func import (
    anchor_center,
)
from multitask_perception.modeling.heads.detection.nanodet.utils.helper_func import (
    anchor_inside_flags,
)
from multitask_perception.modeling.heads.detection.nanodet.utils.helper_func import (
    images_to_levels,
)
from multitask_perception.modeling.heads.detection.nanodet.utils.helper_func import (
    multi_apply,
)
from multitask_perception.modeling.heads.detection.nanodet.utils.helper_func import (
    reduce_mean,
)
from multitask_perception.modeling.heads.detection.nanodet.utils.helper_func import unmap
from multitask_perception.modeling.heads.detection.nanodet.utils.integral import Integral
from multitask_perception.modeling.heads.detection.nanodet.utils.pseudo_sampler import (
    PseudoSampler,
)


class NanoDetLoss(nn.Module):
    def __init__(
        self,
        cfg,
        anchor_ratios=[1.0],
        anchor_base_sizes=None,
    ):
        super().__init__()
        self.img_size = (cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE)
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.use_sigmoid_cls = True
        self.cls_out_channels = cfg.MODEL.NUM_CLASSES
        self.reg_max = cfg.MODEL.HEAD.REG_MAX
        self.anchor_strides = cfg.MODEL.HEAD.STRIDES
        self.distribution_project = Integral(self.reg_max)
        octave_base_scale = cfg.MODEL.LOSS.OCTAVE_BASE_SCALE
        scales_per_octave = cfg.MODEL.LOSS.SCALES_PER_OCTAVE

        self.loss_qfl = QualityFocalLoss(use_sigmoid=True, beta=2.0, loss_weight=1.0)
        self.loss_dfl = DistributionFocalLoss(loss_weight=0.25)
        self.loss_bbox = GIoULoss(loss_weight=2.0)

        self.anchor_base_sizes = (
            list(cfg.MODEL.HEAD.STRIDES)
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

    def loss_single(
        self,
        anchors,
        cls_score,
        bbox_pred,
        labels,
        label_weights,
        bbox_targets,
        stride,
        num_total_samples,
    ):  # 7
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = torch.nonzero(
            (labels >= 0) & (labels < bg_class_ind), as_tuple=False
        ).squeeze(1)

        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[
                pos_inds
            ]  # (n, 4 * (reg_max + 1))  ！！！！！NAN？？？？？？
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = anchor_center(pos_anchors) / stride

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.distribution_project(pos_bbox_pred)
            pos_decode_bbox_pred = distance2bbox(
                pos_anchor_centers, pos_bbox_pred_corners
            )
            pos_decode_bbox_targets = pos_bbox_targets / stride
            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True
            )
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = bbox2distance(
                pos_anchor_centers, pos_decode_bbox_targets, self.reg_max
            ).reshape(-1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0,
            )

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0,
            )
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = (
                torch.tensor(0).cuda() if torch.cuda.is_available() else torch.tensor(0)
            )

        # qfl loss
        loss_qfl = self.loss_qfl(
            cls_score,
            (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples,
        )

        return loss_qfl, loss_bbox, loss_dfl, weight_targets.sum()

    def forward(self, preds, gt_labels, gt_bboxes):  # 1
        cls_scores, bbox_preds = preds

        input_height, input_width = self.img_size
        img_shapes = [
            [input_height, input_width] for i in range(cls_scores[0].shape[0])
        ]
        gt_bboxes_ignore = None

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_shapes, device=device
        )  # "img_shape":shape of the image input to the network as a tuple(h, w, c)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.gfl_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_shapes,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
        )
        if cls_reg_targets is None:
            return None

        (
            anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets

        num_total_samples = (
            reduce_mean(torch.tensor(num_total_pos).cuda()).item()
            if torch.cuda.is_available()
            else reduce_mean(torch.tensor(num_total_pos)).item()
        )
        num_total_samples = max(num_total_samples, 1.0)

        losses_qfl, losses_bbox, losses_dfl, avg_factor = multi_apply(
            self.loss_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            self.anchor_strides,
            num_total_samples=num_total_samples,
        )

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).item()
        if avg_factor <= 0:
            loss_qfl = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
            loss_bbox = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
            loss_dfl = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
        else:
            losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
            losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))

            loss_qfl = sum(losses_qfl)
            loss_bbox = sum(losses_bbox)
            loss_dfl = sum(losses_dfl)

        loss = loss_qfl + loss_bbox + loss_dfl
        loss_states = dict(loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)

        return loss, loss_states

    def get_anchors(self, featmap_sizes, img_shapes, device="cuda"):  # 2 # checked!
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_shapes (h,w): Image meta info.
            device (torch.device | str): device for returned tensors

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_shapes)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i], device=device
            )
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_shape in enumerate(img_shapes):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w = img_shape
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w), device=device
                )
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def gfl_target(
        self,
        anchor_list,
        valid_flag_list,
        gt_bboxes_list,
        img_shape_list,
        gt_bboxes_ignore_list=None,
        gt_labels_list=None,
        label_channels=1,
        unmap_outputs=True,
    ):  # 3
        """
        almost the same with anchor_target, with a little modification,
        here we need return the anchor
        """
        num_imgs = len(img_shape_list)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self.gfl_target_single,
            anchor_list,
            valid_flag_list,
            num_level_anchors_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_shape_list,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs,
        )
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        return (
            anchors_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def gfl_target_single(
        self,
        flat_anchors,
        valid_flags,
        num_level_anchors,
        gt_bboxes,
        gt_bboxes_ignore,
        gt_labels,
        img_shape,
        label_channels=1,
        unmap_outputs=True,
    ):  # 4
        device = flat_anchors.device
        gt_bboxes = torch.from_numpy(gt_bboxes).to(device)
        gt_labels = torch.from_numpy(gt_labels).to(device)
        # num_gts = gt_labels.size(0)
        # if num_gts > 0:
        #     gt_labels += 1

        inside_flags = anchor_inside_flags(
            flat_anchors, valid_flags, img_shape, allowed_border=-1
        )
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags
        )
        bbox_assigner = ATSSAssigner(topk=9)
        assign_result = bbox_assigner.assign(
            anchors, num_level_anchors_inside, gt_bboxes, gt_bboxes_ignore, gt_labels
        )

        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors, gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full(
            (num_valid_anchors,), self.num_classes, dtype=torch.long
        )
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            # if cfg.pos_weight <= 0:
            #     label_weights[pos_inds] = 1.0
            # else:
            #     label_weights[pos_inds] = cfg.pos_weight
            label_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes
            )
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (
            anchors,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            pos_inds,
            neg_inds,
        )

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):  # 5
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [int(flags.sum()) for flags in split_inside_flags]
        return num_level_anchors_inside
