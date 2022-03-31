from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset, replace_ImageToTensor)
import argparse
from mmdet import __version__
from mmcv.utils import get_git_hash
from mmcv.runner import GradientCumulativeOptimizerHook
parser = argparse.ArgumentParser()


parser.add_argument('--cfg_file', type=str, default='./atss_swinL_fpn_dyhead_1x_trash.py', help='config_file_path')
parser.add_argument('--exp_name', type=str, default='faster_rcnn_r50_fpn_1x_trash', help='experiment name')
parser.add_argument('--train_resize', default=[(512,512), (768, 768), (1024, 1024)], help='train_resize')
parser.add_argument('--test_resize', default=[(512,512), (768, 768), (1024, 1024)], help='test_resize')
parser.add_argument('--samples_per_gpu', type=int, default=2, help='samples_per_gpu')
parser.add_argument('--seed', type=int, default=2022, help='seed')
parser.add_argument('--workflow', default=[('train', 1)], help='workflow')
parser.add_argument('--valid', default=True, help='validation')
parser.add_argument('--max_epochs', default=20, help='epochs')
parser.add_argument('--resume_from', default=None, help='resume_from_pthfile_path')

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

# model
# cfg.model.roi_head.bbox_head.num_classes = 10
# optimizer

# workflow
cfg.workflow = args.workflow
cfg.runner.max_epochs = args.max_epochs
cfg.resume_from = args.resume_from

# meta
meta = dict()
meta['seed'] = cfg.seed
# meta['exp_name'] = osp.basename(args.config)

#print(DATASETS)

# Train
# build_dataset
datasets = [build_dataset(cfg.data.train)]

if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        #val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

# cfg.optim_hook = GradientCumulativeOptimizerHook(cumulative_iters=16)

# meta
if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)

# 모델 build 및 pretrained network 불러오기
model = build_detector(cfg.model)
model.init_weights()

# 모델 학습 (Build data loaders, MMDataParallel, build runner(include model, optimizer, logger, workdir, register runner's hook, valid dataloader,...), runner run)
train_detector(model, datasets, cfg, distributed=False, validate=args.valid, meta=meta)

