# optimizer
optimizer = dict(type='Adam', lr=5e-5)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=4300,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=0.1
    )

runner = dict(type='EpochBasedRunner', max_epochs=12)

