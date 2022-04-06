import os
import numpy as np
import pandas as pd
import argparse
import json

from tqdm import tqdm, trange

import copy

DATA_PATH = '/opt/ml/detection/dataset'
TRAIN_JSON_PATH = "/opt/ml/detection/dataset/stratified_kfold/cv_train_1.json"
TEST_JSON_PATH = "/opt/ml/detection/dataset/test.json"

with open(TEST_JSON_PATH, 'r') as f:
    json_data_test = json.load(f)

def main(args):
    BEST_CSV_PATH = args.target
    SAVING_PATH = args.path
    mode = args.mode
    confidence_threshold = args.confidence_threshold
    
    pseudo = pd.read_csv(BEST_CSV_PATH)
    annotation_dict_list = []
    img_dict_list = copy.deepcopy(json_data_test['images'])

    if mode == 2:
        annotation_id = 23144
        # annotation_dict_list 채우기
        for i in trange(len(pseudo)):
            # image_dict_list 채우기 (id 변환)
            img_id = pseudo['image_id'][i]
            img_id = int(img_id[-8:-4]) + 4883
            img_dict_list[i]['id'] = img_id
            
            string = pseudo['PredictionString'][i].strip()
            split_list = string.split()
            
            # confidence score 상위 10개만 뽑음
            boundary_conf_list = split_list[1::6]
            boundary_conf_list = boundary_conf_list[:10]
            boundary_conf = float(boundary_conf_list[-1])
            for j in range(0,len(split_list)-6, 6):
                if float(split_list[j+1]) < boundary_conf:
                    break
                    
                annotation_dict = {}
                annotation_dict['image_id'] = img_id
                
                
                category = int(split_list[j])
                boxes = [float(split_list[j+k]) for k in range(2, 6)]
                boxes[2] -= boxes[0]
                boxes[3] -= boxes[1]
                
                annotation_dict['category_id'] = category
                annotation_dict['area'] = boxes[2] * boxes[3]
                annotation_dict['bbox'] = boxes
                annotation_dict['iscrowd'] = 0
                annotation_dict['id'] = annotation_id
                annotation_id += 1
                annotation_dict_list.append(annotation_dict)
    else:
        annotation_id = 23144
        delete_idx = []
        print(len(pseudo))
        # annotation_dict_list 채우기
        for i in trange(len(pseudo)):
            # image_dict_list 채우기 (id 변환)
            img_id = pseudo['image_id'][i]
            img_id = int(img_id[-8:-4]) + 4883
            img_dict_list[i]['id'] = img_id
            
            string = pseudo['PredictionString'][i].strip()
            split_list = string.split()
            
            num = 0
            # confidence score threshold 넘는 것만 살리기
            for j in range(0,len(split_list)-6, 6):
                if float(split_list[j+1]) > confidence_threshold:
                    num += 1              
                    annotation_dict = {}
                    annotation_dict['image_id'] = img_id
                    
                    
                    category = int(split_list[j])
                    boxes = [float(split_list[j+k]) for k in range(2, 6)]
                    boxes[2] -= boxes[0]
                    boxes[3] -= boxes[1]
                    
                    annotation_dict['category_id'] = category
                    annotation_dict['area'] = boxes[2] * boxes[3]
                    annotation_dict['bbox'] = boxes
                    annotation_dict['iscrowd'] = 0
                    annotation_dict['id'] = annotation_id
                    annotation_id += 1
                    annotation_dict_list.append(annotation_dict)
            if num == 0:
                delete_idx.append(i)
                
        del_ind = 0
        for i in delete_idx:
            del img_dict_list[(i - del_ind)]
            del_ind += 1
            
    with open(TRAIN_JSON_PATH, 'r') as feedsjson:
        feeds = json.load(feedsjson)

    feeds['images'].extend(img_dict_list)
    feeds['annotations'].extend(annotation_dict_list)

    SAVE_DIR = os.path.join(SAVING_PATH, 'pseudo_labeling.json')
    with open(SAVE_DIR, mode='w') as f:
        f.write(json.dumps(feeds, indent=2))
    print("Done!!")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, default='/opt/ml/detection/dataset', help='pseudo_labeling.json saving path')
    parser.add_argument('--target', '-t', type=str, default="/opt/ml/detection/baseline/mmdetection/level2-object-detection-level2-cv-08/Ensemble/atss_dyhead+cascade_rcnn+yolov5_061_06.csv", help='target csv that will be on pseudo labeling')
    parser.add_argument('--mode', '-m', type=int, default=1, help='mode2: extract top 10 confidence score bbox / mode1: cutting by confidence score')
    parser.add_argument('--confidence_threshold', '-c', type=float, default=0.58, help='In mode2, we need confidence score to be threshold')
    arg = parser.parse_args()
    main(arg)