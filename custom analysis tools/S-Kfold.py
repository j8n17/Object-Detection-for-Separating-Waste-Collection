from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from pycocotools.coco import COCO
import os
import json
import logging
import copy
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate stratified Kfolded train/val annotation files')
    
    parser.add_argument(
        '--ann-path',
        default='/opt/ml/detection/dataset/train_revision.json',
        help='original train annotation path')
    parser.add_argument(
        '--kfold',
        type=int,
        default=8,
        help='number of k in kfold')
    parser.add_argument(
        '--save-dir',
        default='/opt/ml/detection/dataset',
        help='new annotation files save directory')

    
    # parser.add_argument(
    #     '--tp-iou-thr',
    #     type=float,
    #     default=0.5,
    #     help='IoU threshold to be considered as matched')
    # parser.add_argument(
    #     '--nms-iou-thr',
    #     type=float,
    #     default=None,
    #     help='nms IoU threshold, only applied when users want to change the'
    #     'nms IoU threshold.')
    # parser.add_argument(
    #     '--cfg-options',
    #     nargs='+',
    #     action=DictAction,
    #     help='override some settings in the used config, the key-value pair '
    #     'in xxx=yyy format will be merged into config file. If the value to '
    #     'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
    #     'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
    #     'Note that the quotation marks are necessary and that no white space '
    #     'is allowed.')
    args = parser.parse_args()
    return args

def setLog():
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)

    logger.info(f'\n============= Make Stratified KFold =============\n')

    return logger

def getClsPerImg(coco, ids, len_img):
    cls_per_img = np.zeros((len_img, 10))
    for id in ids:
        ann_id = coco.getAnnIds(imgIds=id)
        ann_list = coco.loadAnns(ann_id)
        for ann in ann_list:
            label = ann['category_id']
            cls_per_img[id][label] += 1
    return cls_per_img

def checkNumPerCls(index, coco):
    check_num = [0 for _ in range(10)]
    for id in index:
        ann_id = coco.getAnnIds(imgIds=id)
        ann_list = coco.loadAnns(ann_id)
        for ann in ann_list:
            label = ann['category_id']
            check_num[label] += 1
    return check_num

def makeAnnotations(coco, ann_path, train_index, logger, args):
    logger.info(f'>>>>>>> start making annotation files.... please wait....')

    with open(os.path.join('/opt/ml/detection/dataset/train.json'), 'r') as read_file:
        json_data = json.load(read_file)
    
    json_data_train = copy.deepcopy(json_data)
    json_data_val = copy.deepcopy(json_data)
    add_train_idx = []
    add_val_idx = []

    json_data_train['images']=[]
    json_data_train['annotations']=[]
    json_data_val['images']=[]
    json_data_val['annotations']=[]

    img_id_train_cnt = 0
    img_id_val_cnt = 0
    img_ann_id_match_train = dict()
    img_ann_id_match_val = dict()
    for i, data in enumerate(json_data['images']):
        temp_info = copy.deepcopy(json_data['images'][i])
        if data['id'] in train_index:
            json_data_train['images'].append(temp_info)
            json_data_train['images'][-1]['id']=img_id_train_cnt
            img_ann_id_match_train[data['id']]=img_id_train_cnt
            img_id_train_cnt += 1
            add_train_idx.append(i)
        else:
            json_data_val['images'].append(temp_info)
            json_data_val['images'][-1]['id']=img_id_val_cnt
            img_ann_id_match_val[data['id']]=img_id_val_cnt
            img_id_val_cnt += 1
            add_val_idx.append(i)

    ann_id_train_cnt = 0
    ann_id_val_cnt = 0
    for i, data in enumerate(json_data['annotations']):
        temp_anno = copy.deepcopy(json_data['annotations'][i])
        if data['image_id'] in add_train_idx:
            json_data_train['annotations'].append(temp_anno)
            json_data_train['annotations'][-1]['id']=ann_id_train_cnt
            json_data_train['annotations'][-1]['image_id']=img_ann_id_match_train[data['image_id']]
            ann_id_train_cnt += 1
        else:
            json_data_val['annotations'].append(temp_anno)
            json_data_val['annotations'][-1]['id']=ann_id_val_cnt
            json_data_val['annotations'][-1]['image_id']=img_ann_id_match_val[data['image_id']]
            ann_id_val_cnt += 1
    
    train_path = os.path.join(args.save_dir, 'SK_train_annotations.json')
    val_path = os.path.join(args.save_dir, 'SK_val_annotations.json') 
    with open(train_path, 'w', encoding='utf-8') as make_file:
        json.dump(json_data_train, make_file, indent='\t')

    with open(val_path, 'w', encoding='utf-8') as make_file:
        json.dump(json_data_val, make_file, indent='\t')

    logger.info(f'>>>>>>> check your files : {train_path}, {val_path}')

def main():
    args = parse_args()
    logger = setLog()

    coco = COCO(args.ann_path)

    img_id = coco.getImgIds()
    img_info = coco.loadImgs(img_id)
    fnames = [info['file_name'] for info in img_info]
    ids = np.asarray([info['id'] for info in img_info])
    len_img = len(ids)

    cls_per_img = getClsPerImg(coco, ids, len_img)
    mskf = MultilabelStratifiedKFold(n_splits=args.kfold)

    train_index, val_index = list(mskf.split(ids, cls_per_img))[0]
    
    check_train_num = checkNumPerCls(train_index, coco)
    check_val_num = checkNumPerCls(val_index, coco)
    
    logger.info(f'>>>>>>> Train images per class : {check_train_num}')
    logger.info(f'>>>>>>> Validation images per class : {check_val_num}')
    
    makeAnnotations(coco, args.ann_path, train_index, logger, args)


if __name__ == '__main__':
    main()


    