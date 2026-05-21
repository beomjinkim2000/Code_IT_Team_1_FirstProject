"""postprocess skeleton"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch

class Prediction(TypedDict):
  image_id: int
  boxes: torch.Tensor
  scores: torch.Tensor
  labels: torch.Tensor

@dataclass
class PostprocessConfig:
  """후처리 파라미터
  todo(issue #35): 추후 configs/default.yaml에서 값을 읽도록 연결
  """
  conf_threshold: float = 0.5
  iou_threshold: float = 0.7
  max_detections: int = 4

def postprocess_prediction(
    prediction: Prediction,
    config: PostprocessConfig | None = None,
) -> Prediction:
  """1장의 예측 결과 후처리
  - confidence threshold 기준 필터링
  - score 기준 내림차순 정렬
  - max_detections 기준 상위 예측만 유지
  todo(issue # 35): raw model output 형식이 확정되면 nms 추가
  """
  config = config or PostprocessConfig()

  boxes = prediction["boxes"]
  labels = prediction["labels"]
  scores = prediction["scores"]

  keep = scores >= config.conf_threshold 
  #confidence가 낮은 예측은 제출/평가에 쓰지 않기 위해 제거
  boxes = boxes[keep]
  labels = labels[keep]
  scores = scores[keep]
  # scores shape은 [N], boxes shape은 [N, 4], labels shape은 [N]이어야 한다

  if scores.numel() == 0:
    return make_empty_prediction(image_id=prediction["image_id"])
  # threshold를 통과한 예측이 하나도 없어도 인터페이스 형식은 유지해야 한다.
  # downstream 코드가 항상 boxes/labels/scores 키와 tensor shape을 기대하기 때문
  order = torch.argsort(scores, descending=True)
  # confidence가 높은 예측이 앞에 오도록 정렬한다

  order = order[: config.max_detections]
  # 프로젝트 조건상 한 이미지에서 최대 4개 알약만 탐지한다.
  # todo(issue #35): max_detections는 config 파일 값과 연결되면 여기 기본값을 제거

  return {
    "image_id": int(prediction["image_id"]),
    "boxes": boxes[order].to(dtype=torch.float32),
    "scores": scores[order].to(dtype=torch.float32),
    "labels": labels[order].to(dtype=torch.int64),
  # dtype을 고정 interfaces.md 지킨다.
  # boxes는 규칙상 xyxy 좌표다.  
  }
def postprocess_predictions(
    predictions: list[Prediction],
    config: PostprocessConfig | None = None,
) -> list[Prediction]:
  """여러 이미지 예측 한 번에 후처리
  루트 predict.py 또는 evaluation 단계에서는 이미지 여러 장의 예측 결과를
  한 번에 넘길 수 있으므로 list 단위 helper를 둔다.
  """
  config = config or PostprocessConfig()

  return [
    postprocess_prediction(prediction=prediction, config=config)
    for prediction in predictions
  ]

def make_empty_prediction(image_id: int) -> Prediction:
  """빈 prediction 생성
  예측 결과가 0인 이미지 가능- 형식에 맞게 빈 tesnsor 형식 유지
  """
  return {
    "image_id": int(image_id),
    "boxes": torch.empty((0, 4), dtype=torch.float32),
    "labels": torch.empty((0,), dtype=torch.int64),
    "scores": torch.empty((0,), dtype=torch.float32),
  }