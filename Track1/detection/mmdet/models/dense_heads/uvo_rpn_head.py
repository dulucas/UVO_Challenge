# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import copy
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict
from mmcv.cnn import ConvModule, Scale
from mmcv.ops import DeformConv2d, batched_nms
from mmcv.runner import BaseModule, ModuleList, force_fp32

from mmdet.core import (build_assigner, build_sampler,
                        images_to_levels, multi_apply, anchor_inside_flags,
                        multiclass_nms, unmap, reduce_mean, bbox_overlaps)
from mmdet.core.bbox.assigners.rpn_sim_ota_assigner import RPN_SimOTAAssigner
from ..builder import HEADS, build_head, build_loss
from .base_dense_head import BaseDenseHead
from .rpn_head import RPNHead
from mmdet.core import bbox_mapping_back, merge_aug_proposals, multiclass_nms
if sys.version_info >= (3, 7):
    from mmdet.utils.contextmanagers import completed


class AdaptiveConv(BaseModule):
    """AdaptiveConv used to adapt the sampling location with the anchors.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the conv kernel. Default: 3
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 1
        dilation (int or tuple, optional): Spacing between kernel elements.
            Default: 3
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If set True, adds a learnable bias to the
            output. Default: False.
        type (str, optional): Type of adaptive conv, can be either 'offset'
            (arbitrary anchors) or 'dilation' (uniform anchor).
            Default: 'dilation'.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=3,
                 groups=1,
                 bias=False,
                 type='dilation',
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv'))):
        super(AdaptiveConv, self).__init__(init_cfg)
        assert type in ['offset', 'dilation']
        self.adapt_type = type

        assert kernel_size == 3, 'Adaptive conv only supports kernels 3'
        if self.adapt_type == 'offset':
            assert stride == 1 and padding == 1 and groups == 1, \
                'Adaptive conv offset mode only supports padding: {1}, ' \
                f'stride: {1}, groups: {1}'
            self.conv = DeformConv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                stride=stride,
                groups=groups,
                bias=bias)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=dilation,
                dilation=dilation)

    def forward(self, x, offset):
        """Forward function."""
        if self.adapt_type == 'offset':
            N, _, H, W = x.shape
            assert offset is not None
            assert H * W == offset.shape[1]
            # reshape [N, NA, 18] to (N, 18, H, W)
            offset = offset.permute(0, 2, 1).reshape(N, -1, H, W)
            offset = offset.contiguous()
            x = self.conv(x, offset)
        else:
            assert offset is None
            x = self.conv(x)
        return x


@HEADS.register_module()
class UVOStageCascadeRPNHead(RPNHead):
    """Stage of CascadeRPNHead.

    Args:
        in_channels (int): Number of channels in the input feature map.
        anchor_generator (dict): anchor generator config.
        adapt_cfg (dict): adaptation config.
        bridged_feature (bool, optional): whether update rpn feature.
            Default: False.
        with_cls (bool, optional): wheather use classification branch.
            Default: True.
        sampling (bool, optional): wheather use sampling. Default: True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 cls_head,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     scales=[8],
                     ratios=[1.0],
                     strides=[4, 8, 16, 32, 64]),
                 loss_iou=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 adapt_cfg=dict(type='dilation', dilation=3),
                 bridged_feature=False,
                 show=False,
                 with_cls=True,
                 sampling=True,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01),
                 dcn_on_last_conv=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 conv_bias='auto',
                 use_tower_convs=False,
                 is_dynamic=False,
                 dynamic_balanced_cls_weights=False,
                 update_iter_interval=None,
                 total_iter_num=None,
                 center_radius_target=None,
                 candidate_topk_target=None,
                 aux_center_radius_target=None,
                 aux_candidate_topk_target=None,
                 **kwargs):
        self.use_tower_convs = use_tower_convs
        self.with_cls = with_cls
        self.show = show
        self.anchor_strides = anchor_generator['strides']
        self.anchor_scales = anchor_generator['scales']
        self.bridged_feature = bridged_feature
        self.adapt_cfg = adapt_cfg
        self.cls_head = cls_head
        self.stacked_convs = 4
        self.dcn_on_last_conv = dcn_on_last_conv
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dynamic_balanced_cls_weights = dynamic_balanced_cls_weights
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        super(UVOStageCascadeRPNHead, self).__init__(
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

        self.is_dynamic = is_dynamic
        if self.is_dynamic:
            self.count_history = 0
            self.total_iter_num = total_iter_num
            self.update_iter_interval = update_iter_interval
            self.center_radius_target = center_radius_target
            self.candidate_topk_target = candidate_topk_target
            self.aux_center_radius_target = aux_center_radius_target
            self.aux_candidate_topk_target = aux_candidate_topk_target
        # override sampling and sampler
        self.sampling = sampling
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            if self.train_cfg.aux_assigner is not None:
                self.aux_assigner = build_assigner(self.train_cfg.aux_assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.loss_iou = build_loss(loss_iou)

        if init_cfg is None:
            self.init_cfg = dict(
                type='Normal', std=0.01, override=[dict(name='rpn_reg')])
            self.init_cfg['override'].append(dict(name='rpn_iou', std=0.01, bias_prob=0.01))
            if self.with_cls:
                self.init_cfg['override'].append(dict(name='rpn_cls', std=0.01, bias_prob=0.01))

    def _init_layers(self):
        """Init layers of a CascadeRPN stage."""
        self.rpn_conv = AdaptiveConv(self.in_channels, self.feat_channels,
                                         **self.adapt_cfg)
        if self.use_tower_convs:
            if self.with_cls:
                self._init_cls_convs()
            self._init_reg_convs()
            self.scales = nn.ModuleList(
                    [Scale(1.0) for _ in self.anchor_strides])
        if self.with_cls:
            self.rpn_cls = nn.Conv2d(self.feat_channels,
                                     self.num_anchors * self.cls_out_channels,
                                     1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)
        self.relu = nn.ReLU(inplace=True)
        self.rpn_iou = nn.Conv2d(
            self.feat_channels, self.num_anchors * 1, 3, padding=1)

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def forward_single(self, x, offset, scale):
        """Forward function of single scale."""
        bridged_x = x
        x = self.relu(self.rpn_conv(x, offset))
        if self.bridged_feature:
            bridged_x = x

        if not self.use_tower_convs:
            cls_score = self.rpn_cls(x) if self.with_cls else None
            iou = self.rpn_iou(x)
            bbox_pred = self.rpn_reg(x)
        else:
            cls_feat = x
            reg_feat = x

            if self.with_cls:
                for cls_layer in self.cls_convs:
                    cls_feat = cls_layer(cls_feat)

            for reg_layer in self.reg_convs:
                reg_feat = reg_layer(reg_feat)

            cls_score = self.rpn_cls(cls_feat) if self.with_cls else None
            iou = self.rpn_iou(reg_feat)
            bbox_pred = scale(self.rpn_reg(reg_feat)).float()
        return bridged_x, cls_score, iou, bbox_pred

    def forward_single_wo_scale(self, x, offset):
        """Forward function of single scale."""
        bridged_x = x
        x = self.relu(self.rpn_conv(x, offset))
        if self.bridged_feature:
            bridged_x = x

        cls_score = self.rpn_cls(x) if self.with_cls else None
        iou = self.rpn_iou(x)
        bbox_pred = self.rpn_reg(x)
        return bridged_x, cls_score, iou, bbox_pred


    def forward(self, feats, offset_list=None):
        """Forward function."""
        if offset_list is None:
            offset_list = [None for _ in range(len(feats))]
        if self.use_tower_convs:
            return multi_apply(self.forward_single, feats, offset_list, self.scales)
        else:
            return multi_apply(self.forward_single_wo_scale, feats, offset_list)

    def get_aux_targets(self,
                    scores,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):

        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs
        concat_anchor_list = []
        concat_valid_flag_list = []
        concat_score_list = []

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))
            # convert scores from num_lvl_features-num_imgs to num_imgs-num_lvl_features
            # can be viewed as an inverse function of images_to_levels
            tmp = []
            for j in range(len(scores)):
                tmp.append(scores[j][i].reshape(-1,1))
            concat_score_list.append(torch.cat(tmp))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_aux_target_single,
             concat_score_list,
             concat_anchor_list,
             concat_valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def _get_aux_target_single(self,
                           flat_scores,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):

        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        scores = flat_scores[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        if not isinstance(self.aux_assigner, RPN_SimOTAAssigner):
            assign_result = self.aux_assigner.assign(anchors, num_level_anchors_inside,
                                                 gt_bboxes, gt_bboxes_ignore,
                                                 gt_labels)
        else:
            if self.with_cls:
                use_sqrt = False
            else:
                use_sqrt = True
            assign_result = self.aux_assigner.assign(scores, anchors, num_level_anchors_inside,
                                                 gt_bboxes, gt_bboxes_ignore,
                                                 gt_labels, use_sqrt=use_sqrt)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if hasattr(self, 'bbox_coder'):
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                # used in VFNetHead
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                if self.dynamic_balanced_cls_weights:
                    label_weights[pos_inds] = float(len(neg_inds)) / float(len(pos_inds))
                else:
                    label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            if self.dynamic_balanced_cls_weights:
                label_weights[neg_inds] = float(len(pos_inds)) / float(len(neg_inds))
            else:
                label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)


    def get_uvo_targets(self,
                    scores,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):

        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs
        concat_anchor_list = []
        concat_valid_flag_list = []
        concat_score_list = []

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))
            # convert scores from num_lvl_features-num_imgs to num_imgs-num_lvl_features
            # can be viewed as an inverse function of images_to_levels
            tmp = []
            for j in range(len(scores)):
                tmp.append(scores[j][i].reshape(-1,1))
            concat_score_list.append(torch.cat(tmp))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_uvo_target_single,
             concat_score_list,
             concat_anchor_list,
             concat_valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def _get_uvo_target_single(self,
                           flat_scores,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):

        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        scores = flat_scores[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        if not isinstance(self.assigner, RPN_SimOTAAssigner):
            assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                                 gt_bboxes, gt_bboxes_ignore,
                                                 gt_labels)
        else:
            if self.with_cls:
                use_sqrt = False
            else:
                use_sqrt = True
            assign_result = self.assigner.assign(scores, anchors, num_level_anchors_inside,
                                                 gt_bboxes, gt_bboxes_ignore,
                                                 gt_labels, use_sqrt=use_sqrt)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if hasattr(self, 'bbox_coder'):
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                # used in VFNetHead
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside

    def get_targets(self,
                    scores,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes,
                    img_metas,
                    featmap_sizes,
                    gt_bboxes_ignore=None,
                    label_channels=1):
        """Compute regression and classification targets for anchors.

        Args:
            anchor_list (list[list]): Multi level anchors of each image.
            valid_flag_list (list[list]): Multi level valid flags of each
                image.
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            featmap_sizes (list[Tensor]): Feature mapsize each level
            gt_bboxes_ignore (list[Tensor]): Ignore bboxes of each images
            label_channels (int): Channel of label.

        Returns:
            cls_reg_targets (tuple)
        """
        if  isinstance(self.assigner, RPN_SimOTAAssigner):
            cls_reg_targets = self.get_uvo_targets(
                scores,
                anchor_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                gt_bboxes_ignore_list=gt_bboxes_ignore,
                label_channels=label_channels)
            if self.train_cfg.aux_assigner is not None:
                aux_cls_reg_targets = self.get_aux_targets(
                    scores,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes,
                    img_metas,
                    gt_bboxes_ignore_list=gt_bboxes_ignore,
                    label_channels=label_channels)
        else:
            cls_reg_targets = super(UVOStageCascadeRPNHead, self).get_targets(
                anchor_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                gt_bboxes_ignore_list=gt_bboxes_ignore,
                label_channels=label_channels)
        if self.train_cfg.aux_assigner is not None:
            return cls_reg_targets, aux_cls_reg_targets
        return cls_reg_targets

    def anchor_offset(self, anchor_list, anchor_strides, featmap_sizes):
        """ Get offest for deformable conv based on anchor shape
        NOTE: currently support deformable kernel_size=3 and dilation=1

        Args:
            anchor_list (list[list[tensor])): [NI, NLVL, NA, 4] list of
                multi-level anchors
            anchor_strides (list[int]): anchor stride of each level

        Returns:
            offset_list (list[tensor]): [NLVL, NA, 2, 18]: offset of DeformConv
                kernel.
        """

        def _shape_offset(anchors, stride, ks=3, dilation=1):
            # currently support kernel_size=3 and dilation=1
            assert ks == 3 and dilation == 1
            pad = (ks - 1) // 2
            idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
            yy, xx = torch.meshgrid(idx, idx)  # return order matters
            xx = xx.reshape(-1)
            yy = yy.reshape(-1)
            w = (anchors[:, 2] - anchors[:, 0]) / stride
            h = (anchors[:, 3] - anchors[:, 1]) / stride
            w = w / (ks - 1) - dilation
            h = h / (ks - 1) - dilation
            offset_x = w[:, None] * xx  # (NA, ks**2)
            offset_y = h[:, None] * yy  # (NA, ks**2)
            return offset_x, offset_y

        def _ctr_offset(anchors, stride, featmap_size):
            feat_h, feat_w = featmap_size
            assert len(anchors) == feat_h * feat_w

            x = (anchors[:, 0] + anchors[:, 2]) * 0.5
            y = (anchors[:, 1] + anchors[:, 3]) * 0.5
            # compute centers on feature map
            x = x / stride
            y = y / stride
            # compute predefine centers
            xx = torch.arange(0, feat_w, device=anchors.device)
            yy = torch.arange(0, feat_h, device=anchors.device)
            yy, xx = torch.meshgrid(yy, xx)
            xx = xx.reshape(-1).type_as(x)
            yy = yy.reshape(-1).type_as(y)

            offset_x = x - xx  # (NA, )
            offset_y = y - yy  # (NA, )
            return offset_x, offset_y

        num_imgs = len(anchor_list)
        num_lvls = len(anchor_list[0])
        dtype = anchor_list[0][0].dtype
        device = anchor_list[0][0].device
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        offset_list = []
        for i in range(num_imgs):
            mlvl_offset = []
            for lvl in range(num_lvls):
                c_offset_x, c_offset_y = _ctr_offset(anchor_list[i][lvl],
                                                     anchor_strides[lvl],
                                                     featmap_sizes[lvl])
                s_offset_x, s_offset_y = _shape_offset(anchor_list[i][lvl],
                                                       anchor_strides[lvl])

                # offset = ctr_offset + shape_offset
                offset_x = s_offset_x + c_offset_x[:, None]
                offset_y = s_offset_y + c_offset_y[:, None]

                # offset order (y0, x0, y1, x2, .., y8, x8, y9, x9)
                offset = torch.stack([offset_y, offset_x], dim=-1)
                offset = offset.reshape(offset.size(0), -1)  # [NA, 2*ks**2]
                mlvl_offset.append(offset)
            offset_list.append(torch.cat(mlvl_offset))  # [totalNA, 2*ks**2]
        offset_list = images_to_levels(offset_list, num_level_anchors)
        return offset_list

    def loss_aux_single(self, anchors,
                            cls_score, bbox_pred, iou,
                            labels, label_weights, bbox_targets,
                            aux_labels, aux_label_weights, aux_bbox_targets,
                            num_total_samples, aux_num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            num_total_samples (int): Number os positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        anchors = anchors.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        iou = iou.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        aux_bbox_targets = aux_bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        aux_labels = aux_labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        aux_label_weights = aux_label_weights.reshape(-1)

        # classification loss
        if self.with_cls:
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(
                -1, self.cls_out_channels).contiguous()
            loss_cls = self.loss_cls(
                cls_score, labels, label_weights, avg_factor=num_total_samples)
        else:
            loss_cls = None

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((aux_labels >= 0)
                    & (aux_labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = aux_bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_iou = iou[pos_inds]

            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred)
            pos_decode_bbox_targets = self.bbox_coder.decode(
                pos_anchors, pos_bbox_targets)
            iou_targets = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets, is_aligned=True)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=iou_targets,
                avg_factor=1.0)

            # iou loss
            loss_iou = self.loss_iou(
                pos_iou,
                iou_targets,
                avg_factor=aux_num_total_samples)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_iou = iou.sum() * 0
            iou_targets = aux_bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, loss_iou, iou_targets.sum()


    def loss_single(self, anchors, cls_score, bbox_pred, iou, labels,
                    label_weights, bbox_targets, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            num_total_samples (int): Number os positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        anchors = anchors.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        iou = iou.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # classification loss
        if self.with_cls:
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(
                -1, self.cls_out_channels).contiguous()
            loss_cls = self.loss_cls(
                cls_score, labels, label_weights, avg_factor=num_total_samples)
        else:
            loss_cls = None

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_iou = iou[pos_inds]

            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred)
            pos_decode_bbox_targets = self.bbox_coder.decode(
                pos_anchors, pos_bbox_targets)
            iou_targets = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets, is_aligned=True)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=iou_targets,
                avg_factor=1.0)

            # iou loss
            loss_iou = self.loss_iou(
                pos_iou,
                iou_targets,
                avg_factor=num_total_samples)
            if self.show:
                print('iou_targets:')
                print(iou_targets)
                print(' ')

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_iou = iou.sum() * 0
            iou_targets = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, loss_iou, iou_targets.sum()

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'ious'))
    def loss(self,
             anchor_list,
             valid_flag_list,
             cls_scores,
             ious,
             bbox_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            anchor_list (list[list]): Multi level anchors of each image.
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in bbox_preds]
        device = ious[0].device
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        scores = []
        for i in range(len(cls_scores)):
            if self.with_cls:
                score = cls_scores[i].detach().sigmoid() * \
                        ious[i].detach().sigmoid()
            else:
                score = ious[i].detach().sigmoid()
            scores.append(score)

        if self.train_cfg.aux_assigner is not None:
            cls_reg_targets, aux_cls_reg_targets = self.get_targets(
                scores,
                anchor_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                featmap_sizes,
                gt_bboxes_ignore=gt_bboxes_ignore,
                label_channels=label_channels)
        else:
            cls_reg_targets = self.get_targets(
                scores,
                anchor_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                featmap_sizes,
                gt_bboxes_ignore=gt_bboxes_ignore,
                label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        if isinstance(self.assigner, RPN_SimOTAAssigner):
            (anchor_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
            num_total_pos, num_total_neg) = cls_reg_targets
            if self.train_cfg.aux_assigner is not None:
                (_, aux_labels_list, aux_label_weights_list, aux_bbox_targets_list, aux_bbox_weights_list,
                aux_num_total_pos, aux_num_total_neg) = aux_cls_reg_targets
        else:
            (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
            num_total_pos, num_total_neg) = cls_reg_targets

        if self.sampling:
            num_total_samples = num_total_pos + num_total_neg
        elif isinstance(self.assigner, RPN_SimOTAAssigner):
            num_total_samples = reduce_mean(
                torch.tensor(num_total_pos, dtype=torch.float,
                             device=device)).item()
            num_total_samples = max(num_total_samples, 1.0)
            if self.train_cfg.aux_assigner is not None:
                aux_num_total_samples = reduce_mean(
                    torch.tensor(aux_num_total_pos, dtype=torch.float,
                                 device=device)).item()
                aux_num_total_samples = max(aux_num_total_samples, 1.0)
        else:
            # 200 is hard-coded average factor,
            # which follows guided anchoring.
            num_total_samples = sum([label.numel()
                                     for label in labels_list]) / 200.0

        mlvl_anchor_list = anchor_list

        if self.train_cfg.aux_assigner is not None:
            losses_cls, losses_bbox, losses_iou,\
            bbox_avg_factor = multi_apply(
                self.loss_aux_single,
                mlvl_anchor_list,
                cls_scores,
                bbox_preds,
                ious,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                aux_labels_list,
                aux_label_weights_list,
                aux_bbox_targets_list,
                num_total_samples=num_total_samples,
                aux_num_total_samples=aux_num_total_samples)
            bbox_avg_factor = sum(bbox_avg_factor)
            bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
            losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        else:
            losses_cls, losses_bbox, losses_iou,\
            bbox_avg_factor = multi_apply(
                self.loss_single,
                mlvl_anchor_list,
                cls_scores,
                bbox_preds,
                ious,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                num_total_samples=num_total_samples)

            bbox_avg_factor = sum(bbox_avg_factor)
            bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
            losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        if self.with_cls:
            return dict(
                loss_rpn_cls=losses_cls,
                loss_rpn_reg=losses_bbox,
                loss_rpn_iou=losses_iou)

        if self.is_dynamic:
            self.count_history += 1
            if self.count_history % self.update_iter_interval == 0:
                new_center_radius, new_topk, new_aux_center_radius, new_aux_topk = self.update_hyperparameters(self.count_history)
        return dict(loss_rpn_reg=losses_bbox, loss_rpn_iou=losses_iou)

    def refine_bboxes(self, anchor_list, bbox_preds, img_metas):
        """Refine bboxes through stages."""
        num_levels = len(bbox_preds)
        new_anchor_list = []
        for img_id in range(len(img_metas)):
            mlvl_anchors = []
            for i in range(num_levels):
                bbox_pred = bbox_preds[i][img_id].detach()
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
                img_shape = img_metas[img_id]['img_shape']
                bboxes = self.bbox_coder.decode(anchor_list[img_id][i],
                                                bbox_pred, img_shape)
                mlvl_anchors.append(bboxes)
            new_anchor_list.append(mlvl_anchors)
        return new_anchor_list

    def get_bboxes(self,
                   anchor_list,
                   cls_scores,
                   bbox_preds,
                   ious,
                   img_metas,
                   cfg,
                   rescale=False):
        """Get proposal predict."""
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            iou_list = [
                ious[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list, iou_list,
                                                anchor_list[img_id], img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    # TODO: temporary plan
    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           ious,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for each scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
        Returns:
            Tensor: Labeled boxes have the shape of (n,5), where the
                first 4 columns are bounding box positions
                (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            rpn_iou = ious[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            rpn_iou = rpn_iou.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                rpn_iou = rpn_iou.reshape(-1)
                scores = torch.sqrt(rpn_cls_score.sigmoid() * rpn_iou.sigmoid())
                #scores = rpn_iou.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                if torch.onnx.is_in_onnx_export():
                    # sort op will be converted to TopK in onnx
                    # and k<=3480 in TensorRT
                    _, topk_inds = scores.topk(cfg.nms_pre)
                    scores = scores[topk_inds]
                else:
                    ranked_scores, rank_inds = scores.sort(descending=True)
                    topk_inds = rank_inds[:cfg.nms_pre]
                    scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        # Skip nonzero op while exporting to ONNX
        if cfg.min_bbox_size >= 0 and (not torch.onnx.is_in_onnx_export()):
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]

        # deprecate arguments warning
        if 'nms' not in cfg or 'max_num' in cfg or 'nms_thr' in cfg:
            warnings.warn(
                'In rpn_proposal or test_cfg, '
                'nms_thr has been moved to a dict named nms as '
                'iou_threshold, max_num has been renamed as max_per_img, '
                'name of original arguments and the way to specify '
                'iou_threshold of NMS will be deprecated.')
        if 'nms' not in cfg:
            cfg.nms = ConfigDict(dict(type='nms', iou_threshold=cfg.nms_thr))
        if 'max_num' in cfg:
            if 'max_per_img' in cfg:
                assert cfg.max_num == cfg.max_per_img, f'You ' \
                    f'set max_num and ' \
                    f'max_per_img at the same time, but get {cfg.max_num} ' \
                    f'and {cfg.max_per_img} respectively' \
                    'Please delete max_num which will be deprecated.'
            else:
                cfg.max_per_img = cfg.max_num
        if 'nms_thr' in cfg:
            assert cfg.nms.iou_threshold == cfg.nms_thr, f'You set' \
                f' iou_threshold in nms and ' \
                f'nms_thr at the same time, but get' \
                f' {cfg.nms.iou_threshold} and {cfg.nms_thr}' \
                f' respectively. Please delete the nms_thr ' \
                f'which will be deprecated.'

        dets, keep = batched_nms(proposals, scores, ids, cfg.nms)
        return dets[:cfg.max_per_img]

    def update_hyperparameters(self, curr_iter):

        new_center_radius = self.train_cfg.assigner.center_radius + curr_iter / self.total_iter_num * (self.center_radius_target - self.train_cfg.assigner.center_radius)
        new_topk_candidate = self.train_cfg.assigner.candidate_topk + int(curr_iter / self.total_iter_num * (self.candidate_topk_target - self.train_cfg.assigner.candidate_topk))
        new_aux_center_radius = self.train_cfg.aux_assigner.center_radius + curr_iter / self.total_iter_num * (self.aux_center_radius_target - self.train_cfg.aux_assigner.center_radius)
        new_aux_topk_candidate = self.train_cfg.aux_assigner.candidate_topk + int(curr_iter / self.total_iter_num * (self.aux_candidate_topk_target - self.train_cfg.aux_assigner.candidate_topk))
        if curr_iter / self.total_iter_num > 1:
            new_center_radius = self.center_radius_target
            new_topk_candidate = self.candidate_topk_target
            new_aux_center_radius = self.aux_center_radius_target
            new_aux_topk_candidate = self.aux_candidate_topk_target

        self.assigner.center_radius = new_center_radius
        self.assigner.candidate_topk = new_topk_candidate
        self.aux_assigner.center_radius = new_aux_center_radius
        self.aux_assigner.candidate_topk = new_aux_topk_candidate
        return new_center_radius, new_topk_candidate, new_aux_center_radius, new_aux_topk_candidate

@HEADS.register_module()
class UVORPNHead(BaseDenseHead):
    """The CascadeRPNHead will predict more accurate region proposals, which is
    required for two-stage detectors (such as Fast/Faster R-CNN). CascadeRPN
    consists of a sequence of RPNStage to progressively improve the accuracy of
    the detected proposals.

    More details can be found in ``https://arxiv.org/abs/1909.06720``.

    Args:
        num_stages (int): number of CascadeRPN stages.
        stages (list[dict]): list of configs to build the stages.
        train_cfg (list[dict]): list of configs at training time each stage.
        test_cfg (dict): config at testing time.
    """

    def __init__(self, num_stages, stages, train_cfg, test_cfg, init_cfg=None):
        super(UVORPNHead, self).__init__(init_cfg)
        assert num_stages == len(stages)
        self.num_stages = num_stages
        # Be careful! Pretrained weights cannot be loaded when use
        # nn.ModuleList
        self.stages = ModuleList()
        for i in range(len(stages)):
            train_cfg_i = train_cfg[i] if train_cfg is not None else None
            stages[i].update(train_cfg=train_cfg_i)
            stages[i].update(test_cfg=test_cfg)
            self.stages.append(build_head(stages[i]))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def loss(self):
        """loss() is implemented in UVOStageCascadeRPNHead."""
        pass

    def get_bboxes(self):
        """get_bboxes() is implemented in UVOStageCascadeRPNHead."""
        pass

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None):
        """Forward train function."""
        assert gt_labels is None, 'RPN does not require gt_labels'

        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        device = x[0].device
        anchor_list, valid_flag_list = self.stages[0].get_anchors(
            featmap_sizes, img_metas, device=device)

        losses = dict()

        for i in range(self.num_stages):
            stage = self.stages[i]

            if stage.adapt_cfg['type'] == 'offset':
                offset_list = stage.anchor_offset(anchor_list,
                                                  stage.anchor_strides,
                                                  featmap_sizes)
            else:
                offset_list = None
            x, cls_score, iou, bbox_pred = stage(x, offset_list)
            rpn_loss_inputs = (anchor_list, valid_flag_list, cls_score, iou,
                               bbox_pred, gt_bboxes, img_metas)
            stage_loss = stage.loss(*rpn_loss_inputs)
            for name, value in stage_loss.items():
                losses['s{}.{}'.format(i, name)] = value

            # refine boxes
            if i < self.num_stages - 1:
                anchor_list = stage.refine_bboxes(anchor_list, bbox_pred,
                                                  img_metas)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.stages[-1].get_bboxes(anchor_list, cls_score, bbox_pred, iou,
                                                       img_metas,
                                                       self.test_cfg)
            return losses, proposal_list

    def simple_test_rpn(self, x, img_metas):
        """Simple forward test function."""
        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        device = x[0].device
        anchor_list, _ = self.stages[0].get_anchors(
            featmap_sizes, img_metas, device=device)

        for i in range(self.num_stages):
            stage = self.stages[i]
            if stage.adapt_cfg['type'] == 'offset':
                offset_list = stage.anchor_offset(anchor_list,
                                                  stage.anchor_strides,
                                                  featmap_sizes)
            else:
                offset_list = None
            x, cls_score, iou, bbox_pred = stage(x, offset_list)
            if i < self.num_stages - 1:
                anchor_list = stage.refine_bboxes(anchor_list, bbox_pred,
                                                  img_metas)

        proposal_list = self.stages[-1].get_bboxes(anchor_list, cls_score, bbox_pred, iou,
                                                   img_metas,
                                                   self.test_cfg)
        return proposal_list

    def aug_test_rpn(self, feats, img_metas):
        """Test with augmentation for only for ``RPNHead`` and its variants,
        e.g., ``GARPNHead``, etc.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                        a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Proposals of each image, each item has shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
        """
        scale_ranges = self.test_cfg.fusion_cfg.scale_ranges
        num_same_scale_tta = len(feats) // len(scale_ranges)

        samples_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(samples_per_gpu)]
        for aug_idx, (x, img_meta) in enumerate(zip(feats, img_metas)):
            proposal_list = self.simple_test_rpn(x, img_meta)
            for i, proposals in enumerate(proposal_list):
                min_scale, max_scale = scale_ranges[aug_idx // num_same_scale_tta]
                in_range_idxs = self.remove_boxes(proposals, min_scale, max_scale)
                proposals_new = proposals[in_range_idxs, :]
                aug_proposals[i].append(proposals_new)
                #aug_proposals[i].append(proposals)
        # reorganize the order of 'img_metas' to match the dimensions
        # of 'aug_proposals'
        aug_img_metas = []
        for i in range(samples_per_gpu):
            aug_img_meta = []
            for j in range(len(img_metas)):
                aug_img_meta.append(img_metas[j][i])
            aug_img_metas.append(aug_img_meta)
        # after merging, proposals will be rescaled to the original image size
        merged_proposals = [
            merge_aug_proposals(proposals, aug_img_meta, self.test_cfg)
            for proposals, aug_img_meta in zip(aug_proposals, aug_img_metas)
        ]
        return merged_proposals

    if sys.version_info >= (3, 7):

        async def async_simple_test_rpn(self, x, img_metas):
            sleep_interval = self.test_cfg.pop('async_sleep_interval', 0.025)
            async with completed(
                    __name__, 'rpn_head_forward',
                    sleep_interval=sleep_interval):
                rpn_outs = self(x)

            proposal_list = self.get_bboxes(*rpn_outs, img_metas)
            return proposal_list

    def merge_aug_vote_results(self, aug_bboxes, img_metas):
        """Merge augmented detection bboxes and labels.
        Args:
            aug_bboxes (list[Tensor]): shape (n, 5)
            img_metas (list[Tensor]): shape (3, ).
        Returns:
            tuple: (bboxes, )
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            bboxes[:, :4] = bbox_mapping_back(bboxes[:, :4], img_shape,
                                              scale_factor, flip,
                                              flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        return bboxes

    def remove_boxes(self, boxes, min_scale, max_scale):
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        in_range_idxs = torch.nonzero(
            (areas >= min_scale * min_scale) &
            (areas <= max_scale * max_scale),
            as_tuple=False).squeeze(1)
        return in_range_idxs


