"""Kaggle 제출 CSV 생성 유틸리티.

postprocess.py의 출력(pred_dict 리스트)을 Kaggle 제출 형식으로 변환한다.
내부 bbox 형식은 xyxy 절대 픽셀을 유지하고, 이 파일에서만 제출용 xywh로 바꾼다.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TypedDict

import torch

from src.utils.bbox import xyxy_to_xywh


SUBMISSION_COLUMNS = [
    "annotation_id",
    "image_id",
    "category_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "score",
]


class Prediction(TypedDict):
    """postprocess.py가 반환하는 이미지 1장 단위 예측 결과."""

    image_id: int
    boxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor


def predictions_to_rows(
    predictions: list[Prediction],
    label_to_category: dict[int, int],
    start_annotation_id: int = 1,
) -> list[dict[str, int | float]]:
    """pred_dict 리스트를 submission.csv 행 리스트로 변환한다.

    label_to_category: cfg["data"]["label_to_category"] — class_id → Kaggle category_id 매핑
    """
    rows: list[dict[str, int | float]] = []
    annotation_id = start_annotation_id

    for prediction in predictions:
        image_id = int(prediction["image_id"])

        # postprocess 출력은 xyxy이고, Kaggle 제출은 xywh라서 여기서만 변환한다.
        boxes_xywh = xyxy_to_xywh(prediction["boxes"]).detach().cpu()
        labels = prediction["labels"].detach().cpu()
        scores = prediction["scores"].detach().cpu()

        for box, label, score in zip(boxes_xywh, labels, scores):
            x, y, w, h = box.tolist()
            class_id = int(label.item())
            rows.append(
                {
                    "annotation_id": annotation_id,
                    "image_id": image_id,
                    "category_id": label_to_category[class_id],
                    "bbox_x": float(x),
                    "bbox_y": float(y),
                    "bbox_w": float(w),
                    "bbox_h": float(h),
                    "score": float(score.item()),
                }
            )
            annotation_id += 1

    return rows


def make_submission(
    predictions: list[Prediction],
    label_to_category: dict[int, int],
    output_path: str | Path = "outputs/submissions/submission_v1.csv",
) -> Path:
    """pred_dict 리스트를 Kaggle submission.csv로 저장한다."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = predictions_to_rows(predictions, label_to_category)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return output_path
