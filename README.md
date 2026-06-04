# 경구약제 이미지 인식 | Health Eat AI Team

알약 이미지에서 최대 4개 알약의 **클래스(이름)와 바운딩 박스**를 검출하는 객체 탐지 프로젝트.

---

## 팀 소개

| 역할 | 정 (주담당) | 부 (보조) | 담당 모듈 |
|---|---|---|---|
| 팀장/PM | beomjinkim2000 | - | 아키텍처, `interfaces.md`, `utils/bbox.py`, PR 리뷰, 전체 관리 및 총괄 |
| Data·Dataset 담당 | zipdid | YuJY9897·cjkj1234 | 데이터 확인, annotation 분석, EDA 노트북, `src/data/dataset.py`, `transforms.py`, `split.py` |
| Model·Train 담당 | YuJY9897 | cjkj1234 | `src/models/baseline.py`, `src/engine/train.py`, `evaluate.py` |
| 후처리 담당 | cjkj1234 | YuJY9897 | `src/engine/predict.py`, `submission/` |
| 보고서 | 전원 | - | 실험 로그, 보고서, 발표자료 |

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
│   └── utils/             # bbox.py, seed.py, config.py, validate.py, collate.py
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


## Kaggle 제출

```bash
# 예측 실행 후 submission.csv 생성
python -m src.submission.make_submission --checkpoint outputs/checkpoints/best.pt

# 결과 위치
outputs/submissions/submission_v1.csv
```

---

## 발표 자료

> **GitHub Pages** → **https://beomjinkim2000.github.io/Code_IT_Team_1_FirstProject/slides/**

| 자료 | 링크 |
|---|---|
| 최종 발표 슬라이드 | [slides/health-eat-final.html](https://beomjinkim2000.github.io/Code_IT_Team_1_FirstProject/slides/health-eat-final.html) |
| WandB 실험 리포트 | [slides/health-eat-wandb-report.html](https://beomjinkim2000.github.io/Code_IT_Team_1_FirstProject/slides/health-eat-wandb-report.html) |

Best Score **0.956** · YOLO11x · 3-Phase 학습 · WBF 앙상블 · GrabCut 합성 데이터

---

## 팀 문서

| 항목 | 링크 |
|---|---|
| 📒 팀 협업 일지 (Quartz) | [beomjinkim2000.github.io/Sprint_First_Proj](https://beomjinkim2000.github.io/Sprint_First_Proj/) |
| 👤 김범진 (PM) | [협업일지/김범진(PM)](https://beomjinkim2000.github.io/Sprint_First_Proj/%ED%98%91%EC%97%85%EC%9D%BC%EC%A7%80/%EA%B9%80%EB%B2%94%EC%A7%84(PM)/) |
| 👤 황원재 (Data) | [협업일지/황원재(Data)](https://beomjinkim2000.github.io/Sprint_First_Proj/%ED%98%91%EC%97%85%EC%9D%BC%EC%A7%80/%ED%99%A9%EC%9B%90%EC%9E%AC(Data)/) |
| 👤 유재열 (Model) | [협업일지/유재열(Model)](https://beomjinkim2000.github.io/Sprint_First_Proj/%ED%98%91%EC%97%85%EC%9D%BC%EC%A7%80/%EC%9C%A0%EC%9E%AC%EC%97%B4(Model)/) |
| 👤 박창준 (Exp) | [협업일지/박창준(Exp)](https://beomjinkim2000.github.io/Sprint_First_Proj/%ED%98%91%EC%97%85%EC%9D%BC%EC%A7%80/%EB%B0%95%EC%B0%BD%EC%A4%80(Exp)/) |
| 🔬 WandB 프로젝트 | [health-eat-pill-detection](https://wandb.ai/health-eat-pill-detection/health-eat-pill-detection) |
| 🐙 팀 문서 GitHub | [Sprint_First_Proj](https://github.com/beomjinkim2000/Sprint_First_Proj) |