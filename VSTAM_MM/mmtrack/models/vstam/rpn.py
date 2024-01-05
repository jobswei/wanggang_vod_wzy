import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.core import force_fp32, multiclass_nms
from mmdet.models.builder import  build_loss

from mmdet.models.losses import build_loss
from mmdet.models.builder import build_head, build_bbox_coder, build_sampler
from mmdet.models.losses.iou_loss import BoundedIoULoss
# from mmdet.core.bbox.transforms import delta2bbox

from ..builder import VSTAM
@VSTAM.register_module()
class StandardRPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors, conv_cfg=None, norm_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_anchors = num_anchors

        self.conv = ConvModule(
            in_channels,
            in_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation='relu'
        )

        self.objectness_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.anchor_deltas = nn.Conv2d(in_channels, num_anchors * 4, 1)  # Assuming box_dim is 4

    def forward(self, x):
        t = self.conv(x)
        pred_objectness_logits = self.objectness_logits(t)
        pred_anchor_deltas = self.anchor_deltas(t)
        return [pred_objectness_logits], [pred_anchor_deltas]

@VSTAM.register_module()
class RPN(nn.Module):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 anchor_scales=[8, 16, 32],
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[4, 8, 16, 32, 64],
                 anchor_base_sizes=None,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3,
                 min_pos_iou=0.3,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=[0.0, 0.0, 0.0, 0.0],
                     target_stds=[1.0, 1.0, 1.0, 1.0]),
                 loss_bbox=dict(
                     type='IoULoss',
                     reduction='none',
                     loss_weight=1.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='none',
                     loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.num_anchors = len(anchor_ratios) * len(anchor_scales)
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = anchor_base_sizes
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # Build anchor generator
        self.anchor_generator = AnchorGenerator(
            self.anchor_base_sizes,
            anchor_scales,
            anchor_ratios,
            anchor_strides,
            self.num_anchors)

        # Build samplers
        if train_cfg is not None:
            self.sampler = build_sampler(train_cfg, context=self)
        self.iou_calculator = BoundedIoULoss()

        # Build RPN head
        self.rpn_head = build_head(
            dict(type='StandardRPNHead', in_channels=in_channels, num_anchors=self.num_anchors),
            feat_channels)

    def forward(self, feats):
        num_levels = len(feats)
        assert len(self.anchor_strides) == num_levels

        # Generate anchors
        device = feats[0].device
        mlvl_anchors = self.anchor_generator.grid_anchors(
            feats,
            device=device)

        # Calculate proposal scores and bounding box regression
        rpn_outs = self.rpn_head(feats)
        pred_objectness_logits, pred_anchor_deltas = rpn_outs[0][0], rpn_outs[1][0]

        # Generate proposals
        proposals = self._get_proposals(mlvl_anchors, pred_objectness_logits, pred_anchor_deltas)

        return proposals

    def _get_proposals(self, mlvl_anchors, pred_objectness_logits, pred_anchor_deltas):
        num_levels = len(mlvl_anchors)
        num_imgs = len(pred_objectness_logits)

        mlvl_proposals = []
        for i in range(num_levels):
            anchors_i = mlvl_anchors[i].unsqueeze(0).expand(num_imgs, -1, -1)
            objectness_i = pred_objectness_logits[i]

            # Transpose objectness scores to (N, A, H, W)
            objectness_i = objectness_i.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)
            objectness_i = objectness_i.sigmoid()

            anchor_deltas_i = pred_anchor_deltas[i]
            proposals_i = self.bbox_coder.decode(anchors_i, anchor_deltas_i)
            mlvl_proposals.append(proposals_i)

        return mlvl_proposals

    @force_fp32(apply_to=('proposals',))
    def get_bboxes(self, proposals, img_metas, cfg=None, rescale=False):
        num_levels = len(proposals)
        num_imgs = len(img_metas)

        proposal_list = []
        for i in range(num_imgs):
            valid_flag = proposals[i][:, -1] > 0
            proposals[i] = proposals[i][valid_flag]

            if len(proposals[i]) == 0:
                continue

            if self.test_cfg and 'nms_pre' in self.test_cfg:
                keep = nms(proposals[i], self.test_cfg.nms_pre)
                proposals[i] = proposals[i][keep]

            if self.test_cfg and 'nms' in self.test_cfg:
                keep = multiclass_nms(
                    proposals[i], self.test_cfg.nms, self.test_cfg.max_num)
               
