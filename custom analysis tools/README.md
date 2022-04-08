# level2-object-detection-level2-cv-08

Competition에 필요한 다양한 파일들에 대한 간단한 설명과 사용법 정리!!

# 1. Bbox_vis.ipynb

- **사용 방법**

  submission file이 생성되는 inference.ipynb 뒤에 코드 삽입

- **결과**

  <img src="https://user-images.githubusercontent.com/71866756/159278927-57bd4b8f-ef48-4b84-93bd-7102bcc6bdf5.png" alt="image" style="zoom:50%;" />

# 2. kaggle trash dataset modification

- **사용 방법**

  - Step1. kaggle dataset download

    [여기](https://www.kaggle.com/datasets/kneroma/tacotrashdataset?select=data)에서 download 클릭!

    <img src="https://user-images.githubusercontent.com/71866756/160035859-b0c0cf95-4930-4701-a22c-2398ae993f28.png" alt="image" style="zoom:50%;" />

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
  # 필요한 라이브러리 설치
  pip install iterative-stratification
  ```
  
  ```python
  # 실행 명령어
  > python S-Kfold.py --ann-path [원본 train.json파일 경로] --kfold [kfold에서 k]--save-dir [새로운 annotation file 저장 directory] 
  ```
  

# 5. check_all_images

<img src="../../../../AppData/Roaming/Typora/typora-user-images/image-20220408201025713.png" alt="image-20220408201025713" style="zoom:50%;" />

- **사용 방법**

  - description에 수정하고자 하는 내용을 적는다. 
  - double_click을 더블클릭하면 description 내용을 이름으로 갖는 이미지 파일이 생성된다. 

  

# 6. cutmix specific classes

- 특정 클래스에 대해서 cutmix 이미지 생성 (4개의 cropped 된 박스를 하나의 이미지로 합친다.)

# 7. delete_pth

- pth 파일을 한 번에 삭제할 수 있다. 

# 8. cutmix box tape images

- box tape image를 찾아 cropped 후 합친다. 

  > box tape를 최대한 걸렀지만, 나온 결과를 보면서 박스 테이프가 아닌 이미지는 제거해줘야 한다. 

# 9. confusion_matrix_custom

- inference결과를 confusion matrix 형태로 저장

  > 1. 기존에는 비율만 나왔지만, 개수도 나오도록 수정하였다. 
  >
  >    (confusion_absolute_matrix.png로 나온다. )
  >
  > 2. 검출해내지 못한 박스들에 대해서 wrong_bbox.csv형태로 나오도록 수정

