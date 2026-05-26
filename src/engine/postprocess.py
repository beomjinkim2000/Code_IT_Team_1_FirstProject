"""postprocess skeleton
현재 단계에서는 이미 boxes / labels / scores 형태로 정리된 prediction dict를 입력으로 받는다.

TODO(issue #35):
    실제 YOLO raw output 형식이 확정되면 raw output -> Prediction 변환과 NMS를 추가한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch
from torchvision.ops import batched_nms

from src.utils.bbox import cxcywh_to_xyxy

class Prediction(TypedDict):
  image_id: int
  boxes: torch.Tensor
  scores: torch.Tensor
  labels: torch.Tensor
  
RawOutputs = torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]

@dataclass
class PostprocessConfig:
  """후처리 파라미터
  todo(issue #35): 추후 configs/default.yaml에서 값을 읽도록 연결
  """
  conf_threshold: float = 0.5
  iou_threshold: float = 0.7
  max_detections: int = 4
  class_agnostic_nms: bool = False
  clamp_boxes: bool = False
  max_box_area_ratio: float | None = None
  min_box_area_ratio: float | None = None

def make_empty_prediction(
    image_id: int,
    device: torch.device | str | None = None,
) -> Prediction:
  """예측 결과가 없을떄 인터페이스 형식 유지 함수
  downstream 코드가 항상 boxes/labels/scores 키와 tensor shanpe을 기대
  빈 결과도 정해진 형식으로 반환되도록 유지
  """
  return {
    "image_id": int(image_id),
    "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
    "labels": torch.empty((0,), dtype=torch.int64, device=device),
    "scores": torch.empty((0,), dtype=torch.float32, device=device),
  }

def postprocess_prediction(
    prediction: Prediction,
    config: PostprocessConfig | None = None, 
) -> Prediction:
  """ 이미지 한장의 예측결과 후처리
  현재 스켈레톤 단계에서 처리하는 내용:
    - confidence threshold 기준으로 낮은 score 제거
    - score 기준 내림차순 정렬
    - max_detections 기준으로 상위 예측만 유지

    TODO(issue #35):
        실제 raw output 연결 후 torchvision.ops.nms 또는 YOLO 기준 NMS를 추가한다.
  """

  config = config or PostprocessConfig()
  boxes = prediction["boxes"]
  labels = prediction["labels"]
  scores = prediction["scores"]

  keep = scores >= config.conf_threshold

  boxes = boxes[keep]
  labels = labels[keep]
  scores = scores[keep]

  if scores.numel() == 0:
    return make_empty_prediction(
        image_id=prediction["image_id"],
        device=boxes.device,
    )

  order = torch.argsort(scores, descending=True)
  order = order[: config.max_detections]

  return {
      "image_id": int(prediction["image_id"]),
      "boxes": boxes[order].to(dtype=torch.float32),
      "labels": labels[order].to(dtype=torch.int64),
      "scores": scores[order].to(dtype=torch.float32),
    }
  
def postprocess_predictions(
    predictions: list[Prediction],
    config: PostprocessConfig | None = None,
) -> list[Prediction]:
    """여러 이미지의 예측 결과를 list 단위로 후처리한다.

    root predict.py나 evaluate.py에서는 이미지 여러 장의 결과를 한 번에 다룰 수 있으므로,
    단일 이미지용 postprocess_prediction()을 반복 적용하는 helper를 둔다.
    """

    config = config or PostprocessConfig()

    return [
        postprocess_prediction(prediction=prediction, config=config)
        for prediction in predictions
    ]

def _extract_raw_tensor(raw_outputs: RawOutputs) -> torch.Tensor:
  """model forward 결과에서 실제 raw prediction tensor만 꺼낸다.

  YOLOv8 eval 출력은 보통 tuple 형태이며, 첫 번째 값이
  [B, 4 + num_classes, num_anchors] 형식의 tensor다.
  테스트용 DummyModel도 같은 형식을 따른다.
  """
  if isinstance(raw_outputs, torch.Tensor):
    return raw_outputs

  if isinstance(raw_outputs, (tuple, list)) and len(raw_outputs) > 0:
    first_output = raw_outputs[0]
    if isinstance(first_output, torch.Tensor):
      return first_output

  raise TypeError("raw_outputs는 Tensor 또는 Tensor를 첫 번째 값으로 가진 tuple/list여야 합니다.")

def _postprocess_single_raw_output(
    raw_output: torch.Tensor,
    image_id: int,
    config: PostprocessConfig,
) -> Prediction:
  """이미지 1장의 raw output을 Prediction 형식으로 변환하고 후처리

  입력 raw_output shape:
    [4 + num_classes, num_anchors]

  처리 순서:
    1. bbox 좌표와 class score 분리
    2. class score 중 가장 높은 값을 score로 사용
    3. cxcywh -> xyxy 변환
    4. confidence threshold 필터링
    5. class-aware NMS 적용
    6. max_detections 개수 제한
  """
  if raw_output.ndim != 2:
    raise ValueError("이미지 1장 raw_output은 [4 + num_classes, num_anchors] 형태여야 합니다.")

  if raw_output.shape[0] <= 4:
    raise ValueError("raw_output에는 bbox 4개 값과 최소 1개 이상의 class score가 필요합니다.")

  cxcywh_boxes = raw_output[:4, :].transpose(0, 1)
  class_scores = raw_output[4:, :].transpose(0, 1)

  scores, labels = torch.max(class_scores, dim=1)
  boxes = cxcywh_to_xyxy(cxcywh_boxes).to(dtype=torch.float32)
  labels = labels.to(dtype=torch.int64)
  scores = scores.to(dtype=torch.float32)

  keep = scores >= config.conf_threshold
  boxes = boxes[keep]
  labels = labels[keep]
  scores = scores[keep]

  if scores.numel() == 0:
    return make_empty_prediction(image_id=image_id, device=raw_output.device)

  nms_groups = torch.zeros_like(labels) if config.class_agnostic_nms else labels

  keep_indices = batched_nms(
    boxes=boxes,
    scores=scores,
    idxs=nms_groups,
    iou_threshold=config.iou_threshold,
  )
  keep_indices = keep_indices[: config.max_detections]

  return {
    "image_id": int(image_id),
    "boxes": boxes[keep_indices].to(dtype=torch.float32),
    "labels": labels[keep_indices].to(dtype=torch.int64),
    "scores": scores[keep_indices].to(dtype=torch.float32),
  }

def postprocess_raw_outputs(
    raw_outputs: RawOutputs,
    image_ids: list[int] | None = None,
    config: PostprocessConfig | None = None,
) -> list[Prediction]:
  """배치 단위 raw model output을 Prediction 리스트로 변환.

  predict_batch() 출력은 postprocess 전 raw output이고,
  make_submission.py / evaluate.py 쪽은 Prediction 형식을 기대한다.
  이 함수가 두 단계 사이의 연결 역할을 한다.
  """
  config = config or PostprocessConfig()
  raw_tensor = _extract_raw_tensor(raw_outputs)

  if raw_tensor.ndim != 3:
    raise ValueError("raw output은 [B, 4 + num_classes, num_anchors] 형태여야 합니다.")

  batch_size = raw_tensor.shape[0]

  if image_ids is None:
    image_ids = list(range(batch_size))

  if len(image_ids) != batch_size:
    raise ValueError("image_ids 길이는 raw output의 batch size와 같아야 합니다.")

  return [
    _postprocess_single_raw_output(
      raw_output=raw_tensor[batch_idx],
      image_id=image_ids[batch_idx],
      config=config,
    )
    for batch_idx in range(batch_size)
  ]

