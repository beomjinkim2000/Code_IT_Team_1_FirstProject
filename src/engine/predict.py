"""Prediction skeleton"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import torch

class Prediction(TypedDict):
  image_id: int
  boxes: torch.Tensor
  labels: torch.Tensor
  scores: torch.Tensor

@dataclass
class PredictConfig:
  conf_threshold: float = 0.25
  iou_threshold: float = 0.7
  max_detections: int = 4
  device: str | None = None

# todo(issue #18): checkpoint.py에 모델 로딩 함수가 구현되면
# predict 실행부에서 해당 함수를 호출하도록 연결한다.

def predict_images(
    model: Any,
    image_paths: list[str | Path],
    config: PredictConfig | None = None,
)-> list[Prediction]:
  """ 이미지 경로에 대한 예측값을 반환합니다.

  이슈 #24의 경우, interfaces.md와 일치하는 더미 예측값을 반환합니다.
  """
  config = config or PredictConfig()
  predictions: list[Prediction] = []
  for image_id, _ in enumerate(image_paths):
    #실제 구현에서는 image_path를 전처리한 뒤 model inference 결과로 prediction을 생성한다.
    prediction = make_dummy_prediction(image_id=image_id)
    prediction = postprocess_prediction(prediction, config)
    predictions.append(prediction)
  
  return predictions

def make_dummy_prediction(image_id: int) -> Prediction:
  """interfaces.md 섹션 4 형식에 맞춘 임시 예측값을 생성합니다.

  todo(issue #18): 실제 모델 출력 변환 로직으로 교체한다.
  boxes는 프로젝트 내부 규칙에 따라 xyxy 형식입니다."""
  return {
    "image_id": int(image_id),
    "boxes": torch.tensor([[10.0, 20.0, 50.0, 80.0]], dtype=torch.float32),
    "labels": torch.tensor([1], dtype=torch.int64),
    "scores": torch.tensor([0.9], dtype=torch.float32),
  }

def postprocess_prediction(
    prediction: Prediction,
    config: PredictConfig,
) -> Prediction:
  """신뢰도 필터링, 점수 정렬 및 상위 k개 항목 제거를 적용합니다.
  todo(issue #18): 실제 예측 결과에 NMS 또는 YOLO 후처리 로직이 필요하면 여기서 확장한다.
  """
  boxes = prediction["boxes"]
  labels = prediction["labels"]
  scores = prediction["scores"]

  keep = scores >= config.conf_threshold
  boxes = boxes[keep]
  labels = labels[keep]
  scores = scores[keep]

  if scores.numel() == 0:
    return {
      "image_id": int(prediction["image_id"]),
      "boxes": torch.empty((0, 4), dtype=torch.float32),
      "labels": torch.empty((0,), dtype=torch.int64),
      "scores": torch.empty((0,), dtype=torch.float32),
    }
  
  order = torch.argsort(scores, descending=True)
  order = order[: config.max_detections]

  return {
    "image_id": int(prediction["image_id"]),
    "boxes": boxes[order].to(dtype=torch.float32),
    "labels": labels[order].to(dtype=torch.int64),
    "scores": scores[order].to(dtype=torch.float32),
  } 




