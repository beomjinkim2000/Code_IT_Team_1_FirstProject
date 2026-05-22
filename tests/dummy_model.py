"""
테스트용 더미 픽스처 (closes #43).

실제 YOLOv8n 가중치 없이 engine 단위 테스트에서 사용.

  from tests.dummy_model import DummyModel, make_dummy_raw_output, make_dummy_pred_dict

측정된 yolo.model.eval() 출력 형식:
  tuple[Tensor([B, 4+nc, 8400])]
  - [:, :4, :]  cx, cy, w, h  (절대 픽셀)
  - [:, 4:, :]  클래스 확률   (sigmoid 후 0~1)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple

NUM_ANCHORS = 8400   # P3(80×80=6400) + P4(40×40=1600) + P5(20×20=400)
IMG_SIZE    = 640


class DummyModel(nn.Module):
    """
    yolo.model.eval() 출력 형식을 흉내내는 더미 모델.

    train.py, predict.py 단위 테스트용.
    실제 학습 파라미터 없이 forward 시 랜덤 raw output 반환.
    """

    def __init__(self, num_classes: int = 10, img_size: int = IMG_SIZE):
        super().__init__()
        self.num_classes = num_classes
        self.img_size    = img_size
        self._p = nn.Parameter(torch.zeros(1))  # optimizer 등록용 더미 파라미터

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        B = x.shape[0]
        return (make_dummy_raw_output(B, self.num_classes, x.device),)


def make_dummy_raw_output(
    batch_size:  int,
    num_classes: int = 10,
    device:      torch.device = torch.device("cpu"),
    num_anchors: int = NUM_ANCHORS,
    img_size:    int = IMG_SIZE,
) -> torch.Tensor:
    """
    postprocess.py 입력용 raw output 생성.

    shape: [B, 4 + num_classes, num_anchors]
      - [:, :4, :]  cx, cy, w, h  (0~img_size 범위 절대 픽셀)
      - [:, 4:, :]  클래스 확률   (0~1)
    """
    cx = torch.rand(batch_size, 1, num_anchors, device=device) * img_size
    cy = torch.rand(batch_size, 1, num_anchors, device=device) * img_size
    w  = torch.rand(batch_size, 1, num_anchors, device=device) * img_size
    h  = torch.rand(batch_size, 1, num_anchors, device=device) * img_size
    scores = torch.rand(batch_size, num_classes, num_anchors, device=device)
    return torch.cat([cx, cy, w, h, scores], dim=1)


def make_dummy_pred_dict(
    batch_size:      int = 2,
    num_classes:     int = 10,
    num_detections:  int = 3,
    img_size:        int = IMG_SIZE,
    start_image_id:  int = 0,
) -> List[Dict]:
    """
    evaluate.py 입력용 pred_dict 리스트 생성.

    인터페이스 계약서 postprocess 출력 형식:
      image_id : int
      boxes    : Tensor [N, 4], float32, xyxy 절대 픽셀
      labels   : Tensor [N],    int64
      scores   : Tensor [N],    float32
    """
    result = []
    for i in range(batch_size):
        x1 = torch.rand(num_detections) * (img_size - 60)
        y1 = torch.rand(num_detections) * (img_size - 60)
        x2 = (x1 + torch.rand(num_detections) * 50 + 10).clamp(max=img_size)
        y2 = (y1 + torch.rand(num_detections) * 50 + 10).clamp(max=img_size)
        result.append({
            "image_id": start_image_id + i,
            "boxes":    torch.stack([x1, y1, x2, y2], dim=1).float(),
            "labels":   torch.randint(0, num_classes, (num_detections,), dtype=torch.int64),
            "scores":   torch.rand(num_detections).float(),
        })
    return result
