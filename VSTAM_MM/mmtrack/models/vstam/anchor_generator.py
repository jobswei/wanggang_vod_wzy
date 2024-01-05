import math
from typing import List
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmcv.ops import Conv2d
from mmcv.utils import build_norm_layer

from mmdet.core import anchor_generator
from mmdet.models.builder import ANCHOR_GENERATORS
from mmdet.models.builder import build_loss

@ANCHOR_GENERATORS.register_module()
class DefaultAnchorGenerator(anchor_generator.AnchorGenerator, BaseModule):
    """Anchor generator in MMDetection.

    Args:
        scales (list[list[float]] or list[float]): If sizes is list[list[float]],
            sizes[i] is the list of anchor sizes (i.e. sqrt of anchor area) to use for the i-th
            feature map. If sizes is list[float], the sizes are used for all feature maps.
            Anchor sizes are given in absolute lengths in units of the input image; they do not
            dynamically scale if the input image size changes.
        ratios (list[list[float]] or list[float]): list of aspect ratios (i.e. height / width)
            to use for anchors. Same "broadcast" rule for `sizes` applies.
        strides (list[int]): stride of each input feature.
        offset (float): Relative offset between the center of the first anchor and the top-left
            corner of the image. Value has to be in [0, 1).
    """

    def __init__(self,
                 scales,
                 ratios,
                 strides,
                 offset=0.5,
                 octave_base_scale=4,
                 octave_scale=2,
                 scales_per_octave=3,
                 **kwargs):
        super().__init__(scales, ratios, strides, offset)
        self.octave_base_scale = octave_base_scale
        self.octave_scale = octave_scale
        self.scales_per_octave = scales_per_octave

        self.num_features = len(self.strides)
        self.cell_anchors = self._calculate_anchors()

        self.anchor_generator_cfg = dict(
            scales=self.scales,
            ratios=self.ratios,
            strides=self.strides,
            octave_base_scale=self.octave_base_scale,
            octave_scale=self.octave_scale,
            scales_per_octave=self.scales_per_octave,
        )

    def _calculate_anchors(self):
        cell_anchors = super()._calculate_anchors()
        anchor_list = []
        for anchor in cell_anchors:
            anchor_list.append(torch.Tensor(anchor))
        return anchor_list

    def _grid_anchors(self, grid_sizes):
        anchors = super()._grid_anchors(grid_sizes)
        anchor_list = []
        for anchor in anchors:
            anchor_list.append(torch.Tensor(anchor))
        return anchor_list

    def forward(self, featmap_sizes, device='cuda'):
        grid_sizes = [featmap_size[-2:] for featmap_size in featmap_sizes]
        anchor_list = self._grid_anchors(grid_sizes)
        return anchor_list

    def init_weights(self):
        """Initialize weights of the anchor generator."""
        pass

