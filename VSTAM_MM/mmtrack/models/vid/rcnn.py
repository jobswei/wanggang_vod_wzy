import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
# from ..builder import build_backbone
from ..builder import MODELS
from .base import BaseVideoDetector
from mmcv import Config
from detectron2.modeling.proposal_generator.build import build_proposal_generator
from detectron2.modeling.roi_heads.roi_heads import build_roi_heads
from detectron2.config import get_cfg
from detectron2.modeling.backbone.build import build_backbone
@MODELS.register_module()
class GeneralizedRCNN(BaseVideoDetector):
    def __init__(self, backbone, proposal_generator, roi_head,init_cfg=None):
        super(GeneralizedRCNN, self).__init__(init_cfg)
        # self.backbone = build_backbone(backbone)
        # proposal_generator.MODEL=Config()
        # proposal_generator.MODEL.PROPOSAL_GENERATOR=proposal_generator
        elsecfg=get_cfg()
        elsecfg.MODEL.BACKBONE.NAME=backbone.type
        self.backbone = build_backbone(elsecfg)
        self.proposal_generator = build_proposal_generator(elsecfg,self.backbone.output_shape()) # 这里直接使用detectron2库里的rpn和roiHead
        self.roi_head = build_roi_heads(elsecfg, self.backbone.output_shape())

    def init_weights(self):
        pass

    @auto_fp16()
    def forward_train(self, img, img_metas, **kwargs):
        x = self.backbone(img)
        if self.proposal_generator:
            proposals, _ = self.proposal_generator(x)
        else:
            raise NotImplementedError("You need to implement proposal generator")
        _, detector_losses = self.roi_head(x, proposals, img_metas)
        losses = {}
        losses.update(detector_losses)
        return losses

    @auto_fp16()
    def simple_test(self, img, img_metas, **kwargs):
        x = self.backbone(img)
        if self.proposal_generator:
            proposals, _ = self.proposal_generator(x)
        else:
            raise NotImplementedError("You need to implement proposal generator")
        results = self.roi_head.simple_test(x, proposals, img_metas)
        return results

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError

