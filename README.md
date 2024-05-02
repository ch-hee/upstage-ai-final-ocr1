# Receipt Text Detection | 영수증 글자 검출
## OCR 1조

| ![강승현](https://files.slack.com/files-tmb/T05UGFFGL07-F071STU5T3L-2760f45993/____________________________720.jpg) | ![김창희](https://avatars.githubusercontent.com/u/156163982?v=4) | ![문정의](https://files.slack.com/files-tmb/T05UGFFGL07-F071NTSPAKU-d427f15783/kakaotalk_20240502_190837783_720.jpg) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [강승현](https://github.com/kangggggggg)             |            [김창희](https://github.com/ch-hee)             |            [문정의](https://github.com/axa123-moon)             |
|                            👑                             |                                                         |                                                         |


## 0. Overview
### Environment
- NVIDIA GeForce RTX 3090
- CUDA Version 11.8

### Requirements

hydra-core==1.3.2
imageio==2.33.0
lightning==2.1.3
pytorch-lightning==2.1.3
matplotlib==3.8.2
numpy==1.26.2
numba==0.58.1
opencv-python==4.8.1.78
pandas==2.1.4
pathlib==1.0.1
Pillow==10.1.0
Polygon3==3.0.9.1
pyclipper==1.3.0.post5
PyYAML==6.0.1
safetensors==0.4.1
setuptools==69.0.3
scikit-image==0.22.0
scikit-learn==1.3.2
scipy==1.11.4
seaborn==0.13.0
shapely==2.0.2
tensorboard==2.15.1
tensorboard-data-server==0.7.2
timm==0.9.12
torchmetrics==1.2.1
tqdm==4.66.1
wandb==0.16.1
--extra-index-url https://download.pytorch.org/whl/cu118
torch==2.1.2+cu118
--extra-index-url https://download.pytorch.org/whl/cu118
torchvision==0.16.2+cu118

## 1. Competiton Info

### Overview

- 영수증 글자 검출(Receipt Text Detection) 대회는 인공지능 모델을 이용하여 제공된 영수증 이미지에서 문자의 위치를 정확하게 검출하는 문제에 도전하는 대회입니다.

### Timeline

- April 8, 2024 - Start Date
- May 2, 2024 - Final submission deadline

## 2. Components

### Directory

```
├── datasets
│   ├─── images
│   │   ├── train
│   │   │   ├── drp.en_ko.in_house.selectstar_NNNNNN.jpg
│   │   │   ├── ...
│   │   │   └── drp.en_ko.in_house.selectstar_NNNNNN.jpg
│   │   ├── val
│   │   │   ├── drp.en_ko.in_house.selectstar_NNNNNN.jpg
│   │   │   ├── ...
│   │   │   └── drp.en_ko.in_house.selectstar_NNNNNN.jpg
│   │   └── test
│   │   │   ├── drp.en_ko.in_house.selectstar_NNNNNN.jpg
│   │   │   ├── ...
│   │   │   └── drp.en_ko.in_house.selectstar_NNNNNN.jpg
│   └─── jsons
│       ├── train.json
│       ├── val.json
│        └── test.json
└─── configs
    ├── preset
    │   ├── example.yaml
    │   ├── base.yaml
    │   ├── datasets
    │   │   └── db.yaml
    │   ├── lightning_modules
    │   │   └── base.yaml
    │   ├── metrics
    │   │   └── cleval.yaml
    │   └── models
    │       ├── decoder
    │       │   └── unet.yaml
    │       ├── encoder
    │       │   └── timm_backbone.yaml
    │       ├── head
    │       │   └── db_head.yaml
    │       ├── loss
    │       │   └── db_loss.yaml
    │       ├── postprocess
    │       │   └── base.yaml
    │       └── model_example.yaml
    ├── train.yaml
    ├── test.yaml
    └── predict.yaml
```

## 3. Data descrption

### Dataset overview

- **대회 데이터셋 License**
    - [CC-BY-NC](https://creativecommons.org/licenses/by-nc/2.0/kr/deed.ko)

- **(학습 데이터셋)**
    - images/train 디렉토리, images/val 디렉토리에 학습용 영수증 이미지가 저장되어 있습니다.
    - jsons 디렉토리에 train.json, val.json 파일이 저장되어 있습니다.
    - 3,273장의 train 이미지, 404장의 validation 영수증 이미지(train / val 의 구분은 학습의 용이함을 위해 구분되어 있지만, 다른 기준으로 재분류 하거나 validation 셋을 학습에 사용하여도 무방).
      
- **(테스트 데이터셋)**
    - images/test 디렉토리에 평가 데이터용 영수증 이미지가 저장되어 있습니다.
    - jsons 디렉토리에 test.json 파일이 저장되어 있습니다.
    - 학습 데이터와 유사한 조건에서 촬영된 413장의 영수증 이미지. 이미지 당 평균 Words수가 동일하게 분배되어 있습니다.
 
- **JSON 상세 설명**
    - **images**
        - **IMAGE_FILENAME** : 이미지 파일 이름이 경로 없이 저장되어 있습니다.
            - **words**
                - **nnnn** : 이미지마다 검출된 words의 index 번호이며, 0으로 채운 4자리 정수값 입니다. 시작은 1 입니다.
                    - **points**
                        - **List**
                            - X Position, Y Position : 검출한 Text Region의 이미지 상 좌표입니다. 데이터 원본 이미지 기준입니다.
                            - X Position, Y Position : 검출한 Text Region의 이미지 상 좌표입니다. 데이터 원본 이미지 기준입니다.
                            - X Position, Y Position : 검출한 Text Region의 이미지 상 좌표입니다. 데이터 원본 이미지 기준입니다.
                            - X Position, Y Position : 검출한 Text Region의 이미지 상 좌표입니다. 데이터 원본 이미지 기준입니다.
                            - ... : 좌표는 최소 4점 이상 존재해야 평가 대상이 됩니다. 4점 미만의 경우 평가에서 예외처리 됩니다.

- 평가 Metric 본 대회에서는 리더보드 순위를 CLEval Metric을 이용하여 도출한 H-Mean (Higher is better)으로 순위를 결정
    - **CLEval (Character-Level Evaluation for Text Detection and Recognition Tasks)**
        - OCR Task에서 Text Detection을 위한 Character-level 평가 도구
        - 문자 인식 정확도에 중점을 두며, 모델이 얼마나 정확하게 문자를 인식하고 있는지에 대한 평가 기준을 제공합니다(ex. "RIVERSIDE"를 "RIVER" "SIDE"로 검출 하더라도 Text Detection으로는 문제가 없으므로, 이런 유형의 문제를 해결하고자 고안).
    - **H-Mean**
      ![H-Mean](https://github.com/UpstageAILab/upstage-ai-final-ocr1/blob/main/images/hmean.png)   

### EDA

- **Language**
  
    - train Language
      ![train Language](https://github.com/UpstageAILab/upstage-ai-final-ocr1/blob/main/images/1_train_Language.png?raw=true)

    - validation Language
      ![validation Language](https://github.com/UpstageAILab/upstage-ai-final-ocr1/blob/main/images/2_validation_Language.png?raw=true)

    - 대부분은 ['ko'] 인 한국어 이미지(wordbox 내용이 숫자인 경우 포함)
      
- **Orientation**

    - train Orientation 
      ![train Orientation](https://github.com/UpstageAILab/upstage-ai-final-ocr1/blob/main/images/3_train_Orientation.png?raw=true)

    - validation Orientation
      ![validation Orientation](https://github.com/UpstageAILab/upstage-ai-final-ocr1/blob/main/images/4_validation_Orientation.png?raw=true)

    - 대부분 수직(Horizontal) 방향.
      
- **이미지 당 Word box 개수**

    - train wordbox 
      ![train wordbox](https://github.com/UpstageAILab/upstage-ai-final-ocr1/blob/main/images/5_train_wordbox.png?raw=true)
      *train wordbox*

    - validation wordbox
      ![validation wordbox](https://github.com/UpstageAILab/upstage-ai-final-ocr1/blob/main/images/6_validation_wordbox.png?raw=true)

    - 이미지 당 평균 100개의 이상의 word box가 있는 매우 밀도가 높은 데이터.
      
- **Wordbox 밀도 분포**

    - train wordbox density
      ![train wordbox density](https://github.com/UpstageAILab/upstage-ai-final-ocr1/blob/main/images/7_train_density.png?raw=true)

    - validation wordbox density
      ![validation wordbox density](https://github.com/UpstageAILab/upstage-ai-final-ocr1/blob/main/images/8_validation_density.png?raw=true)

    - 중앙에 wordbox 밀집.

### Data Processing

- 배경 제거를 위한 방법
  
    - Rembg
      ![Rembg](https://raw.githubusercontent.com/danielgatis/rembg/master/examples/animal-2.jpg)
        - 이미지에서 배경을 제거하는 라이브러리인 Rembg 사용
    
    - Crop
        - 배경 제거를 위해 gt의 최소점 x_min, y_min 과 최대점 x_max, y_max를 통해 crop_image의 w, h을 구한 뒤 10% 비율을 적용하여 image를 crop합니다.

## 4. Modeling

### Model descrition

- **DBNET**
    - Base 모델
       
- **DBNET++**
    ![DBNET++](https://github.com/UpstageAILab/upstage-ai-final-ocr1/blob/main/images/DBNET++.png)

    - DBNET++는 DBNET의 확장 버전으로, 성능 향상을 위해 추가적인 모듈(Adaptive Scale Fusion Module)을 적용한 모델입니다.
        - **adaptive_scale_fusion.py** 다양한 크기의 텍스트에 대응하기 위해 이미지의 다른 해상도에서 특징을 추출하고 이를 융합하는 방식을 사용합니다.
            - **spatial_attention.py** 주어진 특징 맵에서 중요한 공간적 정보를 강조하는 데 사용됩니다.

### Modeling Process

- April 12, 2024 - Baseline 코드 제출. H-Mean(0.8818)
- April 19, 2024 - encoder backbone을 (resnet18 -> efficientnet_b0) 로 설정했을 때 H-Mean이 증가 **(0.8818 -> 0.9084)**
- April 22, 2024 - db_head.yaml에서 use_polygon : True 로 설정했을 때 H-Mean이 증가 **(0.9084 -> 0.9756)**
- April 24, 2024 - encoder backbone을 convnext_tiny.fb_in22k, epoch을 (10 -> 15), batch_size를 (16 -> 8), box_thresh를 (0.4 -> 0.5) 로 설정했을 때 H-Mean이 증가 **(0.9756 -> 0.9763)**
- April 25, 2024 - DBNet++ 구현. DBNet++ 사용시 H-Mean 증가(0.9763 -> 0.9783)
- April 26, 2024 - Crop 구현. DBNet++에서 encoder backbone을 convnext_base.fb_in22k_ft_in1k_384, box_thresh를 (0.5 -> 0.47), max_candidates를 (300 -> 500), negative_ratio를 (3.0 -> 3.5), thresh_map_loss_weight를 (10.0 -> 12.0) 로 설정했을 때 H-Mean이 증가 **(0.9783 -> 0.9832)**
- April 27, 2024 - Rembg 구현.
- April 29, 2024 - epoch 변화 후 H-Mean 증가 **(0.9832 -> 0.9835)**
- May 1, 2024 - train, validation에서 비정상적이라 판단된 word box 제거(ex. 워터마크, 모자이크 된 곳에 word box가 있는 것, 빈 박스, 글자가 아닌 것, 영수증 밖에 있는 word box, 손가락에 가려진 word box 등).

### summary

#### Model

| 구분     | 모델       | 설명 | Model Stats |
|----------|------------|-----------------------------------------|-------------|
| Backbone | DBNET      | Backbone Base 모델                      |             |
|          | DBNET++    | Adaptive Scale Fusion Module 적용하여 성능 향상. 관련 코드 작성하여 적용함 |             |
| Encoder  | Resnet18   | encoder Base 모델                       | Params (M): 11.7, GMACs: 1.8, Activations (M): 2.5 |
|          | ConvNext   | 가장 성능이 좋게 평가됨                  | Params (M): 28.6, GMACs: 4.5, Activations (M): 13.4 |
|          | Efficient b0 | ConvNext와 근접한 성능 나타냄          | Params (M): 5.3, GMACs: 0.4, Activations (M): 6.7 |
|          | Efficient b5 | ConvNext와 근접한 성능 나타냄          | Params (M): 30.4, GMACs: 10.5, Activations (M): 98.9 |
|          | Efficient v2 | ConvNext와 근접한 성능 나타냄          | Params (M): 8.1, GMACs: 0.8, Activations (M): 4.6 |
| Decoder  | Unet       | decoder Base 모델                       |             |

#### Hyper-parameter tuning 

| 백본    | 인코더 모델     | 하이퍼파라미터 / 변경 작업                                           | H-Mean | Precision | Recall |
|---------|----------------|--------------------------------------------------------------------|--------|-----------|--------|
| DBNET   | Resnet18       | 기본 설정                                                          | 0.8818 | 0.9651    | 0.8194 |
| DBNET   | Efficientnet b0| 기본 설정                                                          | 0.9084 | 0.9665    | 0.8631 |
| DBNET   | Convnext       | 기본 설정                                                          | 0.9084 | 0.9665    | 0.8631 |
| DBNET   | Convnext       | "use_polygon: True <br> box_thresh: 0.5"                           | 0.9756 | 0.9762    | 0.9761 |
| DBNET   | Convnext       | thresh_map_loss_weight: 12.0                                       | 0.9775 | 0.9791    | 0.9767 |
| DBNET++ | Convnext       | crop_image 사용                                                    | 0.9783 | 0.9795    | 0.9782 |
| DBNET++ | Convnext       | "crop_image 사용 <br> box_thresh: 0.47 <br> max_candidates: 500 <br> negative_ratio: 3.5 <br> thresh_map_loss_weight: 12.0" | **0.9835** | 0.9842    | 0.9832 |
| DBNET++ | Convnext       | Train의 word box 수정                                              | 0.9820 | 0.9878    | 0.9767 |

#### Status of implementation of suggestions

| 의견                                            | 구현 유무          | 비고             |
|-------------------------------------------------|-------------------|------------------|
| DBNet ++ 코드 테스트 해보기                      | 창희님 구현 및 적용|                  |
| 전/후처리 모듈 추가하기 - Rembg (배경이미지를 검정색으로 변환) | 승현님 구현 및 적용|                  |
| SOTA 모델로 학습 - TextFuseNet, MixNet          | 미구현            |                  |
| torchvision.Compose의 ToTensor 조언            | 구현              |                  |
| 전/후처리 모듈 추가하기 - thin-plate-spline 알고리즘 | 미구현            |                  |
| 전/후처리 모듈 추가하기 - Equalize historam    | 미적용            |                  |


## 5. Result

### Leader Board

![Public H-mean](https://github.com/UpstageAILab/upstage-ai-final-ocr1/blob/main/images/public.png)  
      
- Public H-mean : 0.9835, 0.9832 **(1)**
  
![Private H-mean](https://github.com/UpstageAILab/upstage-ai-final-ocr1/blob/main/images/private.png) 
        
- Private H-mean : 0.9815 **(1)**

### Presentation

- 

## etc

### Reference

- [CLEval (Character-Level Evaluation for Text Detection and Recognition Tasks)](https://github.com/clovaai/CLEval)
- [Rembg](https://github.com/danielgatis/rembg)
- [DBNET++](https://github.com/MhLiao/DB)
- [Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion](https://paperswithcode.com/paper/real-time-scene-text-detection-with-1)
