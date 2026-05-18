# 경구약제 이미지 인식 | Health Eat AI Team

알약 이미지에서 최대 4개 알약의 **클래스(이름)와 바운딩 박스**를 검출하는 객체 탐지 프로젝트.

---

## 팀 소개

| 역할 | 담당 모듈 |
|---|---|
| 팀장/PM | 아키텍처, `interfaces.md`, `utils/bbox.py`, `submission/`, PR 리뷰 |
| Data 담당 | 데이터 확인, annotation 분석, EDA 노트북 |
| Dataset 담당 | `src/data/dataset.py`, `transforms.py`, `split.py` |
| Model/Train 담당 | `src/models/baseline.py`, `src/engine/train.py`, `evaluate.py` |
| Inference/Report 담당 | `src/engine/predict.py`, 보고서, 발표자료 |

---

## 기술 스택

- Python 3.14 / [uv](https://docs.astral.sh/uv/)
- PyTorch / YOLOv8 (Ultralytics)
- OpenCV, Albumentations, Pandas, Matplotlib

---

## 프로젝트 구조

```
project/
├── interfaces.md          # 팀 인터페이스 계약서 (변경 시 팀장 승인)
├── train.py               # 학습 진입점
├── configs/
│   └── default.yaml       # 하이퍼파라미터 & 경로 설정
├── data/
│   ├── raw/               # AI Hub 원본 (git 추적 제외)
│   └── processed/         # YOLO 포맷 변환 결과
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_visualize_bbox.ipynb
├── src/
│   ├── data/              # dataset.py, transforms.py, split.py
│   ├── models/            # baseline.py (build_model)
│   ├── engine/            # train.py, evaluate.py, predict.py
│   ├── submission/        # make_submission.py
│   └── utils/             # bbox.py, visualize.py, seed.py
├── outputs/               # checkpoints, predictions, submissions (git 추적 제외)
└── reports/               # 실험 로그, 보고서
```

---

## 환경 설정

```bash
# 의존성 설치
uv sync

# 학습 실행
python train.py --config configs/default.yaml
```

---

## 데이터셋

AI Hub 경구약제 이미지 데이터셋 사용. `data/raw/`에 직접 다운로드.

**주의**: 아래 두 데이터셋은 학습에 사용 금지 (Kaggle 테스트셋 기반)
- `경구약제조합 5000종 > TL_2_조합.zip`
- `경구약제조합 5000종 > TS_2_조합.zip`

---

## 실험 결과

| 버전 | 모델 | 변경사항 | val mAP@50 | Kaggle Score |
|---|---|---|---|---|
| v0.1 | YOLOv8n | 파이프라인 첫 완성 | - | - |

---

## Kaggle 제출

```bash
# 예측 실행 후 submission.csv 생성
python -m src.submission.make_submission --checkpoint outputs/checkpoints/best.pt

# 결과 위치
outputs/submissions/submission_v1.csv
```

---

## 협업 일지

| 이름 | 링크 |
|---|---|
| (팀원별 Notion 또는 블로그 링크 추가) | |