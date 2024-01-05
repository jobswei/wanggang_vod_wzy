import warnings
# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry
MODELS = Registry('models', parent=MMCV_MODELS)
TRACKERS = MODELS
MOTION = MODELS
REID = MODELS
AGGREGATORS = MODELS
DETECTORS = MODELS
MEMORY=MODELS
BACKBONES = MODELS
NECKS = MODELS
ROI_EXTRACTORS = MODELS
SHARED_HEADS = MODELS
HEADS = MODELS
LOSSES = MODELS
def build_tracker(cfg):
    """Build tracker."""
    return TRACKERS.build(cfg)


def build_motion(cfg):
    """Build motion model."""
    return MOTION.build(cfg)

def build_memory(cfg):
    """Build neck."""
    return MEMORY.build(cfg)

def build_reid(cfg):
    """Build reid model."""
    return REID.build(cfg)


def build_aggregator(cfg):
    """Build aggregator model."""
    return AGGREGATORS.build(cfg)

def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)

def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)

def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)

def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)

def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model."""
    if train_cfg is None and test_cfg is None:
        return MODELS.build(cfg)
    else:
        return MODELS.build(cfg, MODELS,
                            dict(train_cfg=train_cfg, test_cfg=test_cfg))

def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
