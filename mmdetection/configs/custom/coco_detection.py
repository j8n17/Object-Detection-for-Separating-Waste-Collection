# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/' ## data set 위치
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing") ## class 정의
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True), # Resize
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
        img_scale=(512, 512), # Resize
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg), 
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4, ## gpu당 batch사이즈 몇으로 할건지 , 2->4 
    workers_per_gpu=6, # data loader 를 만들때 worker개수 선언해주는 것과 동일 default =2
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json', ## train annotation file 위치
        img_prefix=data_root, # + 'train2017/', ## data root 위치
        classes = classes, # classes 추가
        pipeline=train_pipeline), ## classes 추가
    val=dict(
        type=dataset_type,
        ann_file='data/coco/' + 'val2017/', # + 'val.json', ## validation annotation file 위치
        img_prefix=data_root, # + 'val2017/', ## data root 위치
        classes = classes,
        pipeline=test_pipeline), ## classes 추가
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json', ## test annotation file 위치
        img_prefix=data_root, # + 'val2017/', ## data root 위치
        classes = classes,
        pipeline=test_pipeline)) ## classes 추가
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50')
