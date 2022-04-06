# dataset settings
dataset_type = 'CocoDataset'
data_root = '../../../dataset'
gpu_ids = [0]
work_dir = './work_dirs'
seed = 2022
resize_scale = [(512, 512)]
img_norm_cfg = dict(
    mean=[109.96, 117.28, 123.46], std=[54.89, 53.50, 54.10], to_rgb=True)
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
mosaic_pipeline = [
    dict(type='Mosaic', img_scale=(1024, 1024)),
    dict(type='Resize', img_scale=resize_scale, multiscale_mode='value', keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=resize_scale, multiscale_mode='value', keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
train_dataset = dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + '/pseudo_labeling.json',
        img_prefix='../../../dataset',
        pipeline=train_pipeline)

mosaic_dataset = dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + '/train.json',
            img_prefix='../../../dataset',
            classes=classes,
            pipeline=[dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
            ]
        ),
        pipeline=mosaic_pipeline)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + '/stratified_kfold/cv_val_1.json',
        img_prefix='../../../dataset',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='../../../dataset/test.json',
        img_prefix='../../../dataset',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP', classwise=True)
