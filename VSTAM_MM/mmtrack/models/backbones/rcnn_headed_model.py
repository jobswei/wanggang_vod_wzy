import torch
from .vstam_backbone import BaseBackbone

# from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
# from mmcv.cnn import build_norm_layer
# from mmcv.ops import DeformConv2d
# from ..builder import BACKBONES
from detectron2.modeling import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, Backbone

@BACKBONE_REGISTRY.register()
class RCNNHeadedModel(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.res4 = BaseBackbone()
        self.out_channels = self.res4.feature_embedder_out

    def forward(self, image, imgs_suplementary=None):
        # Because we can send only one image to detectron2 to default detection
        # so we sent first image and concatenate previous frames here
        if not imgs_suplementary:
            SEQUENCE_LENGTH=0
            imgs_suplementary = torch.ones((image.shape[0], SEQUENCE_LENGTH, image.shape[1], image.shape[2],image.shape[3])).to(image.device)
        concated = []
        for main_img, stacked_img in zip(image, imgs_suplementary):
            concated.append(torch.unsqueeze(torch.cat([
                torch.unsqueeze(main_img, 0), stacked_img], 0), 0))
        concated = torch.cat(concated, 0)
        return {"res4": self.res4(concated)}

    def output_shape(self):
        return {"res4": ShapeSpec(channels=2048, stride=16)}

