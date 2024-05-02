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


### EDA

- _Describe your EDA process and step-by-step conclusion_

### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_
