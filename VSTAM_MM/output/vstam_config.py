optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
checkpoint_config = dict(interval=5000, by_epoch=True)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
model = dict(
    type='GeneralizedRCNN',
    backbone=dict(type='RCNNHeadedModel'),
    proposal_generator=dict(
        type='RPN',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[32, 64, 128, 256, 512],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    roi_head=dict(
        type='Res5ROIHeads',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
            out_channels=256,
            featmap_strides=[16]),
        bbox_head=dict(
            type='',
            num_classes=80,
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))))
dataset_type = 'ImagenetVIDDataset'
data_root = 'data/ILSVRC/'
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqResize', img_scale=(324, 324), keep_ratio=False),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(
        type='SeqNormalize',
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='SeqPad', size_divisor=32),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqResize', img_scale=(328, 328), keep_ratio=False),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
    dict(
        type='SeqNormalize',
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='SeqPad', size_divisor=32),
    dict(
        type='VideoCollect',
        keys=['img'],
        meta_keys=('num_left_ref_imgs', 'frame_stride')),
    dict(type='ConcatVideoReferences'),
    dict(type='MultiImagesToTensor', ref_prefix='ref'),
    dict(type='ToList')
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='ImagenetVIDDataset',
        ann_file='data/ILSVRC/annotations/imagenet_vid_train.json',
        img_prefix='data/ILSVRC/Data/VID',
        ref_img_sampler=dict(
            num_ref_imgs=2,
            frame_range=9,
            filter_key_img=True,
            method='bilateral_uniform'),
        pipeline=[
            dict(type='LoadMultiImagesFromFile'),
            dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
            dict(type='SeqResize', img_scale=(324, 324), keep_ratio=False),
            dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
            dict(
                type='SeqNormalize',
                mean=[102.9801, 115.9465, 122.7717],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='SeqPad', size_divisor=32),
            dict(
                type='VideoCollect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
            dict(type='ConcatVideoReferences'),
            dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
        ]),
    val=dict(
        type='ImagenetVIDDataset',
        ann_file='data/ILSVRC/annotations/imagenet_vid_val.json',
        img_prefix='data/ILSVRC/Data/VID',
        pipeline=[
            dict(type='LoadMultiImagesFromFile'),
            dict(type='SeqResize', img_scale=(328, 328), keep_ratio=False),
            dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
            dict(
                type='SeqNormalize',
                mean=[102.9801, 115.9465, 122.7717],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='SeqPad', size_divisor=32),
            dict(
                type='VideoCollect',
                keys=['img'],
                meta_keys=('num_left_ref_imgs', 'frame_stride')),
            dict(type='ConcatVideoReferences'),
            dict(type='MultiImagesToTensor', ref_prefix='ref'),
            dict(type='ToList')
        ],
        test_mode=True),
    test=dict(
        type='ImagenetVIDDataset',
        ann_file='data/ILSVRC/annotations/imagenet_vid_val.json',
        img_prefix='data/ILSVRC/Data/VID',
        pipeline=[
            dict(type='LoadMultiImagesFromFile'),
            dict(type='SeqResize', img_scale=(328, 328), keep_ratio=False),
            dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
            dict(
                type='SeqNormalize',
                mean=[102.9801, 115.9465, 122.7717],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='SeqPad', size_divisor=32),
            dict(
                type='VideoCollect',
                keys=['img'],
                meta_keys=('num_left_ref_imgs', 'frame_stride')),
            dict(type='ConcatVideoReferences'),
            dict(type='MultiImagesToTensor', ref_prefix='ref'),
            dict(type='ToList')
        ],
        test_mode=True))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[2])
total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=1, metric='bbox')
work_dir = './output'
gpu_ids = [0]
