# level2-object-detection-level2-cv-08

Competition에 필요한 다양한 파일들에 대한 간단한 설명과 사용법 정리!!

# 1. Bbox_vis.ipynb

- **사용 방법**

  submission file이 생성되는 inference.ipynb 뒤에 코드 삽입

- **결과**

  ![image](https://user-images.githubusercontent.com/71866756/159278927-57bd4b8f-ef48-4b84-93bd-7102bcc6bdf5.png)

# 2. kaggle trash dataset modification

- **사용 방법**

  - Step1. kaggle dataset download

    [여기](https://www.kaggle.com/datasets/kneroma/tacotrashdataset?select=data)에서 download 클릭!

    ![image-20220323001024571](../../../AppData/Roaming/Typora/typora-user-images/image-20220323001024571.png)

  - Step2. kaggle_dataset_transform.ipynb download

  - Step3. 실행!

- **kaggle_dataset_transform.ipynb에서 수정해야 하는 부분**

  - base_path 수정

    ```python
    base_path = '다운로드한 데이터 폴더 경로'
    ```

  - 새로 생성할 annotation 파일명 수정

    ```python
    # '수정한 json 저장' section
    # 아래 코드에서 './new_annotations.json'을 원하는 이름으로 수정
    with open('./new_annotations.json', 'w', encoding='utf-8') as make_file:
        json.dump(json_data_modified, make_file, indent='\t')
    ```


# 3. Result Analysis tool

- **사용 방법**

  - **Step1**  

    `/opt/ml/detection/baseline/mmdetection/tools/test.py` 실행하여 결과를 기록한 .pkl파일 생성

    ```python
    # 예시
    > python test.py [config파일경로] [checkpoint 파일 경로] --work-dir [metric을 저장할 폴더 경로] --out [pkl 파일 저장 폴더 경로] 
    
    # 추가옵션 - bbox가 그려진 이미지를 저장하고 싶다면, --show-dir [폴더명] 추가
    ```

    

  - **Step2** 

    `confusion_matrix_custom.py` 실행하여 confusion matrix 확인

    > TP, FN 확인 가능

    

  - **Step3**

    `analysis.ipynb`의 `시각화 섹션` 실행하여 bbox 시각화

    ![image-20220324231223629](../../../AppData/Roaming/Typora/typora-user-images/image-20220324231223629.png)

    > img_idx : 이미지 번호
    >
    > score_thr : Positive로 판단하기 위한 threshold
    >
    > tp_iou_thr : True Positive로 판단하기 위한 threshold
    >
    > cls_num : 시각화하고 싶은 label 번호 (10의 경우 전체 label에 대하여 시각화)

  - **Step4**

    `analysis.ipynb`의 `통계치 섹션` 실행하여 통계치 시각화

# 4. Stratified K-fold

- **사용 방법**

  ```python
  > python S-Kfold.py --ann-path [원본 train.json파일 경로] --kfold [kfold에서 k]--save-dir [새로운 annotation file 저장 directory] 
  ```

  
