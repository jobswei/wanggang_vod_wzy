_base_ = [
    '../../_base_/default_runtime.py',
]
# mmdetection config
# change the model config in .yaml file
model = dict(
    type='GeneralizedRCNN',
    
    backbone=dict(
        type='RCNNHeadedModel',
    ),
    proposal_generator=dict(
        type='RPN',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[32, 64, 128, 256, 512],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
        ),
        loss_bbox=dict(
            type='SmoothL1Loss',
            beta=1.0,
            loss_weight=1.0,
        ),
    ),
    roi_head=dict(
        type='Res5ROIHeads',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
            out_channels=256,
            featmap_strides=[16],
        ),
        bbox_head=dict(
            type='',
            num_classes=80,
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
            ),
            loss_bbox=dict(
                type='SmoothL1Loss',
                beta=1.0,
                loss_weight=1.0,
            ),
        ),
    ),
    
)

# Dataset config

# dataset settings
dataset_type = "ImagenetVIDDataset"
data_root = "data/ILSVRC/"
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type="LoadMultiImagesFromFile"),
    dict(type="SeqLoadAnnotations", with_bbox=True, with_track=True),
    dict(type="SeqResize", img_scale=(324, 324), keep_ratio=False),
    dict(type="SeqRandomFlip", share_params=True, flip_ratio=0.5),
    dict(type="SeqNormalize", **img_norm_cfg),
    dict(type="SeqPad", size_divisor=32),
    dict(
        type="VideoCollect", keys=["img", "gt_bboxes", "gt_labels", "gt_instance_ids"]
    ),
    dict(type="ConcatVideoReferences"),
    dict(type="SeqDefaultFormatBundle", ref_prefix="ref"),
]
test_pipeline = [
    dict(type="LoadMultiImagesFromFile"),
    dict(type="SeqResize", img_scale=(328, 328), keep_ratio=False),
    dict(type="SeqRandomFlip", share_params=True, flip_ratio=0.0),
    dict(type="SeqNormalize", **img_norm_cfg),
    dict(type="SeqPad", size_divisor=32),
    dict(
        type="VideoCollect",
        keys=["img"],
        meta_keys=("num_left_ref_imgs", "frame_stride"),
    ),
    dict(type="ConcatVideoReferences"),
    dict(type="MultiImagesToTensor", ref_prefix="ref"),
    dict(type="ToList"),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
            type=dataset_type,
            ann_file=data_root + "annotations/imagenet_vid_train.json",
            img_prefix=data_root + "Data/VID",
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=9,
                filter_key_img=True,
                method="bilateral_uniform",
            ),
            pipeline=train_pipeline,
        ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/imagenet_vid_val.json',
        img_prefix=data_root + 'Data/VID',
        pipeline=test_pipeline,
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/imagenet_vid_val.json',
        img_prefix=data_root + 'Data/VID',
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

# Training schedule
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=0.001,
#     step=[30000],
#     gamma=0.1,
# )
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=1.0 / 3, step=[2]
)
total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=12)

# Other configurations
checkpoint_config = dict(interval=5000, by_epoch=True)
evaluation = dict(interval=1, metric='bbox')
work_dir = './output'
