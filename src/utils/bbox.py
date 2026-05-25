import torch
from torch import Tensor


def xyxy_to_xywh(boxes: Tensor) -> Tensor:
    """[x1, y1, x2, y2] → [x, y, w, h] (Kaggle 제출용)"""
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)


def xywh_to_xyxy(boxes: Tensor) -> Tensor:
    """[x, y, w, h] → [x1, y1, x2, y2] (학습 내부용)"""
    x, y, w, h = boxes.unbind(dim=-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """[cx, cy, w, h] → [x1, y1, x2, y2] (YOLO raw output → 내부 표준 형식)"""
    cx, cy, w, h = boxes.unbind(dim=-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
