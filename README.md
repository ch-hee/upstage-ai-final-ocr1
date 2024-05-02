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

- _Insert your directory structure_

e.g.
```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
```

## 3. Data descrption

### Dataset overview

- (학습 데이터셋)
    - images/train 디렉토리, images/val 디렉토리에 학습용 영수증 이미지가 저장되어 있습니다.
    - jsons 디렉토리에 train.json, val.json 파일이 저장되어 있습니다.
    - 3,273장의 train 이미지, 404장의 validation 영수증 이미지(train / val 의 구분은 학습의 용이함을 위해 구분되어 있지만, 다른 기준으로 재분류 하거나 validation 셋을 학습에 사용하여도 무방).
    - 
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
      
- Orientation
    - 대부분 수직(Horizontal) 방향.
      
- 이미지 당 Word box 개수
    - 이미지 당 평균 100개의 이상의 word box가 있는 매우 밀도가 높은 데이터.
      
- Wordbox 밀도 분포
    - 중앙에 wordbox 밀집.

### Data Processing

- [Rembg](https://github.com/danielgatis/rembg)
    - 이미지에서 배경을 제거하는 라이브러리인 Rembg 사용

- Crop

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

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
