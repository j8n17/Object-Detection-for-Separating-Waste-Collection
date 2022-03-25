from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset, replace_ImageToTensor)
import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--cfg_file', type=str, default='./faster_rcnn_r50_fpn_1x_trash.py', help='config_file_path')
parser.add_argument('--exp_name', type=str, default='faster_rcnn_r50_fpn_1x_trash', help='experiment name')
parser.add_argument('--train_resize', default=[(512,512), (768, 768), (1024, 1024)], help='train_resize')
parser.add_argument('--test_resize', default=[(512,512), (768, 768), (1024, 1024)], help='test_resize')
parser.add_argument('--samples_per_gpu', type=int, default=8, help='samples_per_gpu')
parser.add_argument('--seed', type=int, default=2022, help='seed')
parser.add_argument('--checkpoint', default=True, help='default: max_keep_ckpts=3, interval=1')
parser.add_argument('--workflow', default=[('train', 1)], help='workflow')
parser.add_argument('--valid', default=False, help='validation')

args = parser.parse_args()

# config file 들고오기
cfg = Config.fromfile(args.cfg_file)

# wandb save config
cfg.log_config.hooks[1].init_kwargs.config = args
# 실험 이름
cfg.log_config.hooks[1].init_kwargs.name = args.exp_name

# dataset config 수정
cfg.data.train.pipeline[2]['img_scale'] = args.train_resize # multi scale resize
cfg.data.test.pipeline[1]['img_scale'] = args.test_resize # Resize
cfg.data.samples_per_gpu = args.samples_per_gpu
cfg.seed = args.seed
cfg.gpu_ids = [0]
cfg.work_dir = './work_dirs'

# model
# cfg.model.roi_head.bbox_head.num_classes = 10
# optimizer
if args.checkpoint:
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)

# workflow
cfg.workflow = args.workflow





#print(DATASETS)

# Train
# build_dataset
datasets = [build_dataset(cfg.data.train)]
#print(type(datasets[0]))

# 모델 build 및 pretrained network 불러오기
model = build_detector(cfg.model)
model.init_weights()

# 모델 학습 (Build data loaders, MMDataParallel, build runner(include model, optimizer, logger, workdir, register runner's hook, valid dataloader,...), runner run)
train_detector(model, datasets[0], cfg, distributed=False, validate=args.valid)

