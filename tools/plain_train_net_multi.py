#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
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

logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            print(data)
            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    # Register Dataset
    try:
        #register_coco_instances('coco_trash_train', {}, '../../dataset/train.json', '../../dataset/')
        register_coco_instances('coco_trash_train', {}, '../../dataset/SK_train_annotations.json', '../../dataset/')
    except AssertionError:
        pass

    try:
        #register_coco_instances('coco_trash_test', {}, '../../dataset/train.json', '../../dataset/')
        register_coco_instances('coco_trash_test', {}, '../../dataset/SK_val_annotations.json', '../../dataset/')
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
    cfg.CUDNN_BENCHMARK = True # 이미지 크기가 모두 같을 때에는 True로 해주는 게 더 좋다.

    '''
    img_scale=[(512, 512), (608, 608),
                        (704, 704), (736, 736),
                        (832, 832), (864, 864),
                        (960, 960), (992, 992),
                        (1056, 1056), (1088, 1088),
                        (1152, 1152), (1184, 1184),
                        (1248, 1248), (1280, 1280),
            ],
            multiscale_mode='value',
            keep_ratio=True),
    '''
    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.2

    cfg.DATASETS.TRAIN = ('coco_trash_train',)
    cfg.DATASETS.TEST = ('coco_trash_test',)

    #cfg.DATALOADER.NUM_WORKERS = 2

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')
    #cfg.MODEL.WEIGHTS = "/opt/ml/detection/baseline/detectron2/detectron2/checkpoint/faster_rcnn_swint_T.pth"
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 42800
    cfg.SOLVER.STEPS = (8000,12000)
    cfg.SOLVER.GAMMA = 0.005
    cfg.SOLVER.CHECKPOINT_PERIOD = 1070
    cfg.SOLVER.AMP.ENABLED = True

    cfg.OUTPUT_DIR = './output'

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # set threshold for this model
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
    # When using pre-trained models in Detectron1 or any MSRA models,
    # std has been absorbed into its conv1 weights, so the std needs to be set 1.
    # Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
    cfg.MODEL.PIXEL_STD = [57.375, 57.120, 58.395]

    cfg.TEST.EVAL_PERIOD = 1070

    # mapper - input data를 어떤 형식으로 return할지 (따라서 augmnentation 등 데이터 전처리 포함 됨)
    import detectron2.data.transforms as T

    def MyMapper(dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict['file_name'], format='BGR')
        
        transform_list = [
            T.Resize((512, 512)),
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
    wandb.run.name = "faster_rcnn_swin_tiny_patch4_window7_224_FPN_3x_detectron2"


    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
