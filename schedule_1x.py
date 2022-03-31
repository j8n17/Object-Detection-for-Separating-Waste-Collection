# optimizer
optimizer = dict(type='Adam', lr=5e-5)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=0.1
    )

runner = dict(type='EpochBasedRunner', max_epochs=12)

