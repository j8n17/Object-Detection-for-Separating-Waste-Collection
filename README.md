# level2-object-detection-level2-cv-08
#### 1. Bbox_vis.ipynb

- **사용법**

  submission file이 생성되는 inference.ipynb 뒤에 코드 삽입

- **submission file을 통한 Bbox 시각화**

  ![image](https://user-images.githubusercontent.com/71866756/159278927-57bd4b8f-ef48-4b84-93bd-7102bcc6bdf5.png)

#### 2. kaggle trash dataset modification

- **사용법**

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

    

  

  
