"""
WandB 워크스페이스 런 테이블 + 비교 패널 세팅
"""
import os
import wandb_workspaces.workspaces as ws
from wandb_workspaces.reports.v2 import (
    BarPlot, LinePlot, ScatterPlot,
    RunComparer, Config as PCConfig, SummaryMetric,
)

ENTITY = os.environ.get("WANDB_ENTITY", "health-eat-pill-detection")
PROJECT = os.environ.get("WANDB_PROJECT", "health-eat-pill-detection")

workspace = ws.Workspace(
    entity=ENTITY,
    project=PROJECT,
    name="실험 비교 뷰",

    runset_settings=ws.RunsetSettings(
        # 런 목록 테이블 고정 컬럼 순서
        pinned_columns=[
            "run:displayName",        # 런 이름
            "config:model",           # 모델명 ← 1순위
            "summary:best_mAP_ema",   # best val mAP (EMA) ← 2순위
            "summary:kaggle_score",   # Kaggle Score ← 3순위
            "summary:total_epochs",   # 실제 완료 에폭 수
            "summary:val/mAP_ema",    # 마지막 epoch val mAP (EMA)
            "summary:val/mAP50_ema",  # val mAP@50 (EMA)
            "config:train.epochs",    # 설정된 목표 epoch 수
            "config:train.batch_size",
        ],
        # kaggle_score 내림차순 정렬 (앙상블 포함 전체 순위)
        order=[
            ws.Ordering(ws.Summary("kaggle_score"), ascending=False),
        ],
    ),

    sections=[
        # ── 섹션 1: 핵심 성능 순위 ────────────────────────────────
        ws.Section(
            name="핵심 성능 순위 (한눈에 보기)",
            is_open=True,
            panels=[
                BarPlot(
                    title="Best val mAP EMA — 런별 순위",
                    metrics=["best_mAP_ema"],
                    orientation="h",
                    max_runs_to_show=20,
                ),
                BarPlot(
                    title="Kaggle Score — 런별 순위",
                    metrics=["kaggle_score"],
                    orientation="h",
                    max_runs_to_show=20,
                ),
            ],
        ),

        # ── 섹션 2: 모델 아키텍처별 비교 ──────────────────────────
        ws.Section(
            name="모델 아키텍처별 성능 비교",
            is_open=True,
            panels=[
                # 아키텍처별 최고 mAP (yolov8n vs yolov8x vs yolo11x)
                BarPlot(
                    title="모델별 Best mAP EMA (max)",
                    metrics=["best_mAP_ema"],
                    groupby="model",
                    groupby_aggfunc="max",
                    groupby_rangefunc="minmax",
                    orientation="h",
                ),
                # 아키텍처별 Kaggle Score 비교
                BarPlot(
                    title="모델별 Kaggle Score (max)",
                    metrics=["kaggle_score"],
                    groupby="model",
                    groupby_aggfunc="max",
                    groupby_rangefunc="minmax",
                    orientation="h",
                ),
                # val mAP vs Kaggle 산점도 — 과적합 확인
                ScatterPlot(
                    title="val mAP (EMA) vs Kaggle Score — 과적합 여부",
                    x=SummaryMetric("best_mAP_ema"),
                    y=SummaryMetric("kaggle_score"),
                    regression=True,
                ),
            ],
        ),

        # ── 섹션 3: 학습 곡선 (epoch-by-epoch) ───────────────────
        ws.Section(
            name="학습 곡선 — epoch별 mAP 변화",
            is_open=True,
            panels=[
                LinePlot(
                    title="val mAP EMA (epoch별)",
                    y=["val/mAP_ema"],
                    smoothing_type="exponential",
                    smoothing_factor=0.6,
                ),
                LinePlot(
                    title="val mAP@50 EMA (epoch별)",
                    y=["val/mAP50_ema"],
                    smoothing_type="exponential",
                    smoothing_factor=0.6,
                ),
                LinePlot(
                    title="Train Loss (epoch별)",
                    y=["train/loss"],
                    smoothing_type="exponential",
                    smoothing_factor=0.6,
                ),
            ],
        ),

        # ── 섹션 4: Epoch/Batch 설정 → 성능 ──────────────────────
        ws.Section(
            name="Epoch 수 → 성능 관계",
            is_open=False,
            panels=[
                ScatterPlot(
                    title="Epoch 수 → Best mAP EMA",
                    x=PCConfig("epochs"),
                    y=SummaryMetric("best_mAP_ema"),
                    regression=True,
                ),
                ScatterPlot(
                    title="Epoch 수 → Kaggle Score",
                    x=PCConfig("epochs"),
                    y=SummaryMetric("kaggle_score"),
                    regression=True,
                ),
            ],
        ),

        # ── 섹션 5: 과적합 분석 ───────────────────────────────────
        ws.Section(
            name="과적합 분석 — val mAP 포화 vs Kaggle 갭",
            is_open=True,
            panels=[
                # val mAP 곡선: 플래토가 눈에 보이는지 확인
                LinePlot(
                    title="val mAP EMA — 포화 구간 확인 (700ep: ep350 이후 +0.0003)",
                    y=["val/mAP_ema"],
                    smoothing_type="exponential",
                    smoothing_factor=0.5,
                ),
                # val mAP 최고치 vs Kaggle 점수 — 갭 시각화
                BarPlot(
                    title="val mAP 최고치 vs Kaggle Score — 갭이 클수록 분포 불일치",
                    metrics=["best_mAP_ema", "kaggle_score"],
                    orientation="h",
                    max_runs_to_show=10,
                ),
                # train loss vs val loss — 둘 다 같이 내려가는지 (고전적 과적합 없음)
                LinePlot(
                    title="Train Loss vs Val Loss — 둘 다 하락 = loss로는 과적합 탐지 불가",
                    y=["loss/train", "loss/val"],
                    smoothing_type="exponential",
                    smoothing_factor=0.6,
                ),
            ],
        ),

        # ── 섹션 6: 런 상세 비교 ──────────────────────────────────
        ws.Section(
            name="런 상세 비교",
            is_open=False,
            panels=[
                RunComparer(diff_only="split"),
            ],
        ),
    ],
)

workspace.save()
print(f"워크스페이스 저장 완료!")
print(f"URL: {workspace.url}")
