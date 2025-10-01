# SAR图像目标检测轻量级模型配置
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

# 模型设置 - 轻量级配置
model = dict(
    type='RotatedFCOS',
    backbone=dict(
        type='ResNet',
        depth=34,  # 使用更轻量级的ResNet34
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet34')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],  # 适配ResNet34的通道数
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='RotatedFCOSHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=2,  # 减少堆叠卷积层数
        feat_channels=128,  # 减少特征通道数
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
        nms_pre=1000,  # 减少预筛选数量
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(
            type='nms_rotated',
            iou_threshold=0.1),
        max_per_img=1000))  # 减少每张图像的最大检测数量

# 数据增强配置 - 简化版
data = dict(
    samples_per_gpu=8,  # 增加批量大小
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
                img_scale=(800, 800),  # 减小图像尺寸
                keep_ratio=True),
            dict(
                type='RandomFlip',
                flip_ratio=0.5),  # 简化翻转增强
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
                img_scale=(800, 800),  # 减小图像尺寸
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
                img_scale=(800, 800),  # 减小图像尺寸
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
    lr=0.01,  # 稍微提高学习率
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
    warmup_iters=300,  # 减少预热迭代次数
    warmup_ratio=0.1,
    step=[6, 9])  # 提前调整学习率

# 运行设置
runner = dict(type='EpochBasedRunner', max_epochs=10)  # 减少训练轮数
checkpoint_config = dict(interval=1)
eval_config = dict(interval=1, metric='mAP')
log_config = dict(
    interval=100,  # 减少日志记录频率
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
work_dir = './work_dirs/sar_detector_light'
seed = 0
gpus = 1
gpu_ids = range(gpus)