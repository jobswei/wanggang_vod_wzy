# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class VSTAM(SingleStageDetector):

    def __init__(self,
                 backbone,
                 proposal_generator,
                 roi_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(VSTAM, self).__init__(backbone, proposal_generator, roi_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)
        
