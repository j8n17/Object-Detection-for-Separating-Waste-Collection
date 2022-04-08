import os
import copy
import torch
import detectron2
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, SimpleTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, build_detection_train_loader

from tools.swin_train_net import SwinTrainer
from detectron2.config import CfgNode as CN

import wandb, yaml

# Register Dataset
try:
    register_coco_instances('coco_trash_train', {}, '../../dataset/train.json', '../../dataset/')
    #register_coco_instances('coco_trash_train', {}, '../../dataset/SK_train_annotations.json', '../../dataset/')
except AssertionError:
    pass

try:
    register_coco_instances('coco_trash_test', {}, '../../dataset/train.json', '../../dataset/')
    #register_coco_instances('coco_trash_test', {}, '../../dataset/SK_val_annotations.json', '../../dataset/')
except AssertionError:
    pass

MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                                                         "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

# config 불러오기
cfg = get_cfg()
'''
#add_swint_config(cfg)
cfg.MODEL.SWINT = CN(new_allowed=True)
cfg.MODEL.SWINT.EMBED_DIM = 96
cfg.MODEL.SWINT.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
cfg.MODEL.SWINT.DEPTHS = [2, 2, 6, 2]
cfg.MODEL.SWINT.NUM_HEADS = [3, 6, 12, 24]
cfg.MODEL.SWINT.WINDOW_SIZE = 7
cfg.MODEL.SWINT.MLP_RATIO = 4
cfg.MODEL.SWINT.DROP_PATH_RATE = 0.2
cfg.MODEL.SWINT.APE = False
cfg.MODEL.BACKBONE.FREEZE_AT = -1

# addation
cfg.MODEL.FPN.TOP_LEVELS = 2
cfg.SOLVER.OPTIMIZER = "AdamW"
'''
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))
#cfg.merge_from_file(model_zoo.get_config_file('SwinT/faster_rcnn_swint_T_FPN_3x.yaml')) #swinT

# config 수정하기
# cfg.CUDNN_BENCHMARK = True # 이미지 크기가 모두 같을 때에는 True로 해주는 게 더 좋다.

#cfg.INPUT.MIN_SIZE_TRAIN = (512, 608, 704, 736, 832, 864, 960, 992, 1056, 1088, 1152, 1184, 1248, 1280)


cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
cfg.DATALOADER.REPEAT_THRESHOLD = 0.2

cfg.DATASETS.TRAIN = ('coco_trash_train',)
cfg.DATASETS.TEST = ('coco_trash_test',)

#cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')
#cfg.MODEL.WEIGHTS = "/opt/ml/detection/baseline/detectron2/detectron2/checkpoint/faster_rcnn_swint_T.pth"
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 21400
cfg.SOLVER.STEPS = (8000,12000)
cfg.SOLVER.GAMMA = 0.005
cfg.SOLVER.CHECKPOINT_PERIOD = 1070
cfg.SOLVER.AMP.ENABLED = False

cfg.OUTPUT_DIR = './output'

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # set threshold for this model
#cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
# When using pre-trained models in Detectron1 or any MSRA models,
# std has been absorbed into its conv1 weights, so the std needs to be set 1.
# Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
#cfg.MODEL.PIXEL_STD = [57.375, 57.120, 58.395]

cfg.TEST.EVAL_PERIOD = 1070

# mapper - input data를 어떤 형식으로 return할지 (따라서 augmnentation 등 데이터 전처리 포함 됨)
import detectron2.data.transforms as T

def MyMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    transform_list = [
        #T.Resize([i for i in range(512, 2048, 192)]),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3)
    ]
    
    image, transforms = T.apply_transform_gens(transform_list, image)
    
    dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))
    
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop('annotations')
        if obj.get('iscrowd', 0) == 0
    ]
    
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict['instances'] = utils.filter_empty_instances(instances)
    
    return dataset_dict

# trainer - DefaultTrainer를 상속
#class MyTrainer(SwinTrainer):
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(
        cfg, mapper = MyMapper, sampler = sampler
        )
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs('./output_eval', exist_ok = True)
            output_folder = './output_eval'
            
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

# train
os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)

wandb.login()
cfg_wandb = yaml.safe_load(cfg.dump())
wandb.init(project="drivingyouth-OD", entity="hbage", config=cfg_wandb)
wandb.run.name = "faster_rcnn_R_101_FPN_3x_detectron2_no_valid"

trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()