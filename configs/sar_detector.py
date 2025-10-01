# SAR图像目标检测模型配置
_base_ = [
    '../mmrotate/configs/_base_/models/fcos_obb_r50_fpn.py',
    '../mmrotate/configs/_base_/datasets/dota1_0_r50_fpn.py',
    '../mmrotate/configs/_base_/schedules/schedule_1x.py',
    '../mmrotate/configs/_base_/default_runtime.py'
]

# 数据集设置
dataset_type = 'SARDataset'
data_root = 'data/sar_dataset/'

# 自定义数据集类
custom_imports = dict(
    imports=['src.dataset'],
    allow_failed_imports=False)

# 类别配置
class_names = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')
num_classes = 6

# 模型设置
model = dict(
    type='RotatedFCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='RotatedFCOSHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='RotatedIoULoss',
            loss_weight=1.0,
            iou_mode='siou'),
        loss_centerness=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(
            type='nms_rotated',
            iou_threshold=0.1),
        max_per_img=2000))

# 数据增强配置
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=class_names,
        ann_file=data_root + 'train.txt',
        img_prefix=data_root + 'images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                img_scale=[(1024, 1024), (1280, 1280), (1536, 1536)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(
                type='RandomFlip',
                flip_ratio=0.5,
                direction=['horizontal', 'vertical']),
            dict(
                type='RandomRotate',
                prob=0.5,
                angle_range=(-10, 10),
                rect_obj_labels=[90, 270]),
            dict(type='Normalize',
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type=dataset_type,
        classes=class_names,
        ann_file=data_root + 'val.txt',
        img_prefix=data_root + 'images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type=dataset_type,
        classes=class_names,
        ann_file=data_root + 'test.txt',
        img_prefix=data_root + 'images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))

# 优化器配置
optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        bias_lr_mult=2.0,
        bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# 学习率调度器
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[8, 11])

# 运行设置
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
eval_config = dict(interval=1, metric='mAP')
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# 环境设置
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/sar_detector'
seed = 0
gpus = 1
gpu_ids = range(gpus)