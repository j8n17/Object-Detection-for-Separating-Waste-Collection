checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1221,
    hooks=[
        dict(type='TextLoggerHook'),
        # wandb 등록
        dict(type='WandbLoggerHook',interval=1000,
            init_kwargs=dict(
                project= "drivingyouth-OD", #'PROJECT 이름',
                entity = "drivingyouth", # 'ENTITY 이름',
                name = "driving" #'실험할때마다 RUN에 찍히는 이름'
            ),
            )
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
# 1 epoch에 train과 validation을 모두 하고 싶으면 workflow = [('train', 1), ('val', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
