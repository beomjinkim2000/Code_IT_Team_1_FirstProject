from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MosaicDataset(Dataset):
    """학습 데이터셋을 래핑해 Mosaic 증강을 적용한다.

    이미지 4장을 2×2 그리드로 합친 뒤 img_size로 리사이즈.
    클래스당 데이터가 적을 때 유효 학습 샘플을 4배로 늘리는 효과.
    """

    def __init__(self, dataset: Dataset, img_size: int, p: float = 0.5, min_bbox_size: int = 2) -> None:
        self.dataset = dataset
        self.img_size = img_size
        self.p = p
        self.min_bbox_size = min_bbox_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        if random.random() > self.p:
            return self.dataset[idx]
        return self._mosaic(idx)

    def _mosaic(self, idx: int):
        s = self.img_size
        indices = [idx] + random.choices(range(len(self.dataset)), k=3)

        canvas = torch.zeros(3, s * 2, s * 2)
        all_boxes: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        # 2×2 그리드: TL / TR / BL / BR 순서
        offsets = [(0, 0), (s, 0), (0, s), (s, s)]  # (x_off, y_off)

        anchor_target = None
        for i, (x_off, y_off) in zip(indices, offsets):
            img, target = self.dataset[i]
            if anchor_target is None:
                anchor_target = target
            canvas[:, y_off:y_off + s, x_off:x_off + s] = img

            boxes = target["boxes"]
            labels = target["labels"]
            if boxes.numel() > 0:
                shifted = boxes.clone()
                shifted[:, [0, 2]] += x_off
                shifted[:, [1, 3]] += y_off
                all_boxes.append(shifted)
                all_labels.append(labels)

        # 2×img_size → img_size 로 리사이즈 (0.5 배율)
        canvas_resized = F.interpolate(
            canvas.unsqueeze(0), size=(s, s), mode="bilinear", align_corners=False
        ).squeeze(0)

        if all_boxes:
            combined_boxes = torch.cat(all_boxes) * 0.5
            combined_labels = torch.cat(all_labels)

            combined_boxes = combined_boxes.clamp(0, s - 1)

            # 리사이즈 후 너무 작아진 박스 제거
            w = combined_boxes[:, 2] - combined_boxes[:, 0]
            h = combined_boxes[:, 3] - combined_boxes[:, 1]
            keep = (w >= self.min_bbox_size) & (h >= self.min_bbox_size)
            combined_boxes = combined_boxes[keep]
            combined_labels = combined_labels[keep]
        else:
            combined_boxes = torch.zeros((0, 4), dtype=torch.float32)
            combined_labels = torch.zeros((0,), dtype=torch.int64)

        return canvas_resized, {
            "boxes": combined_boxes,
            "labels": combined_labels,
            "image_id": int(anchor_target["image_id"]),
            "original_size": (s, s),
        }
