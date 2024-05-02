# Title (Please modify the title)
## Team

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [박패캠](https://github.com/UpstageAILab)             |            [이패캠](https://github.com/UpstageAILab)             |            [최패캠](https://github.com/UpstageAILab)             |            [김패캠](https://github.com/UpstageAILab)             |            [오패캠](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

## 0. Overview
### Environment
- (컴퓨팅 환경) 각자의 RTX 3090 서버를 VSCode와 SSH로 연결하여 사용
- (협업 환경) Github, Wandb
- (의사소통) Slack, Zoom, Google meet

### Requirements
- _Write Requirements_

## 1. Competiton Info

### Overview

- 영수증 글자 검출(Receipt Text Detection) 대회는 인공지능 모델을 이용하여 제공된 영수증 이미지에서 문자의 위치를 정확하게 검출하는 문제에 도전하는 대회입니다

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

- (학습 데이터셋)
    - images/train 디렉토리, images/val 디렉토리에 학습용 영수증 이미지가 저장되어 있습니다.
    - jsons 디렉토리에 train.json, val.json 파일이 저장되어 있습니다.
    - 3,273장의 train 이미지, 404장의 validation 영수증 이미지(train / val 의 구분은 학습의 용이함을 위해 구분되어 있지만, 다른 기준으로 재분류 하거나 validation 셋을 학습에 사용하여도 무방).
      
- (테스트 데이터셋)
    - images/test 디렉토리에 평가 데이터용 영수증 이미지가 저장되어 있습니다.
    - jsons 디렉토리에 test.json 파일이 저장되어 있습니다.
    - 학습 데이터와 유사한 조건에서 촬영된 413장의 영수증 이미지. 이미지 당 평균 Words수가 동일하게 분배되어 있습니다.
 
- JSON 상세 설명
    - images
        - IMAGE_FILENAME : 이미지 파일 이름이 경로 없이 저장되어 있습니다.
            - words
                - nnnn : 이미지마다 검출된 words의 index 번호이며, 0으로 채운 4자리 정수값 입니다. 시작은 1 입니다.
                    - points
                        - List
                            - X Position, Y Position : 검출한 Text Region의 이미지 상 좌표입니다. 데이터 원본 이미지 기준입니다.
                            - X Position, Y Position : 검출한 Text Region의 이미지 상 좌표입니다. 데이터 원본 이미지 기준입니다.
                            - X Position, Y Position : 검출한 Text Region의 이미지 상 좌표입니다. 데이터 원본 이미지 기준입니다.
                            - X Position, Y Position : 검출한 Text Region의 이미지 상 좌표입니다. 데이터 원본 이미지 기준입니다.
                            - ... : 좌표는 최소 4점 이상 존재해야 평가 대상이 됩니다. 4점 미만의 경우 평가에서 예외처리 됩니다.

- 평가 Metric 본 대회에서는 리더보드 순위를 CLEval Metric을 이용하여 도출한 H-Mean (Higher is better)으로 순위를 결정
    - [CLEval (Character-Level Evaluation for Text Detection and Recognition Tasks)](https://github.com/clovaai/CLEval)
        - OCR Task에서 Text Detection을 위한 Character-level 평가 도구
        - 문자 인식 정확도에 중점을 두며, 모델이 얼마나 정확하게 문자를 인식하고 있는지에 대한 평가 기준을 제공(ex. "RIVERSIDE"를 "RIVER" "SIDE"로 검출 하더라도 Text Detection으로는 문제가 없으므로, 이런 유형의 문제를 해결하고자 고안)
    - H-Mean
        -     

### EDA

- Language
    - 대부분은 ['ko'] 인 한국어 이미지(wordbox 내용이 숫자인 경우 포함)
 
      ![train Language](https://github.com/UpstageAILab/upstage-ai-final-ocr1/blob/main/images/1_train_Language.png?raw=true)

      ![validation Language](https://github.com/UpstageAILab/upstage-ai-final-ocr1/blob/main/images/2_validation_Language.png?raw=true)
      
- Orientation
    - 대부분 수직(Horizontal) 방향.
 
      ![train Orientation](https://github.com/UpstageAILab/upstage-ai-final-ocr1/blob/main/images/3_train_Orientation.png?raw=true)

      ![validation Orientation](https://github.com/UpstageAILab/upstage-ai-final-ocr1/blob/main/images/4_validation_Orientation.png?raw=true)
      
- 이미지 당 Word box 개수
    - 이미지 당 평균 100개의 이상의 word box가 있는 매우 밀도가 높은 데이터.
 
      ![train wordbox](https://github.com/UpstageAILab/upstage-ai-final-ocr1/blob/main/images/5_train_wordbox.png?raw=true)

      ![validation wordbox](https://github.com/UpstageAILab/upstage-ai-final-ocr1/blob/main/images/6_validation_wordbox.png?raw=true)
      
- Wordbox 밀도 분포
    - 중앙에 wordbox 밀집.
 
      ![train density](https://github.com/UpstageAILab/upstage-ai-final-ocr1/blob/main/images/7_train_density.png?raw=true)

      ![validation density](https://github.com/UpstageAILab/upstage-ai-final-ocr1/blob/main/images/8_validation_density.png?raw=true)

### Data Processing

- 배경 제거를 위한 방법
  
    - [Rembg](https://github.com/danielgatis/rembg)
        - 이미지에서 배경을 제거하는 라이브러리인 Rembg 사용
    
    - Crop
        - 배경 제거를 위해 gt의 최소점 x_min, y_min 과 최대점 x_max, y_max를 통해 crop_image의 w, h을 구한 뒤 10% 비율을 적용하여 image를 crop합니다.

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

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
| DBNET++ | Convnext       | "crop_image 사용 <br> box_thresh: 0.47 <br> max_candidates: 500 <br> negative_ratio: 3.5 <br> thresh_map_loss_weight: 12.0" | 0.9835 | 0.9842    | 0.9832 |
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

- _Insert Leader Board Capture_
- H-mean : 0.9835

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- [CLEval (Character-Level Evaluation for Text Detection and Recognition Tasks)](https://github.com/clovaai/CLEval)
- [Rembg](https://github.com/danielgatis/rembg)
