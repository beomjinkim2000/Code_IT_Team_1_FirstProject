from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np
import torch
from PIL import Image


class Transform:
    """PIL 이미지와 target dict에 Albumentations 파이프라인을 적용한다."""

    def __init__(self, pipeline: A.Compose) -> None:
        # train/val factory에서 만든 Albumentations 파이프라인을 보관한다.
        self.pipeline = pipeline

    def __call__(self, image: Image.Image, target: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
        # Dataset target 에서 넘어온 bbox/label을 Albumentations 입력 형식으로 정리한다.
        #박스는 애초에 x,4형태로 넘어오고 리사이즈 대비한 텐서작업이 목적
        #라벨은 알약이 라벨이 한 이미지에 3~4개 들어오니까 혹시 모르니 1차원으로 피는작업
        boxes = torch.as_tensor(target["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(target["labels"], dtype=torch.int64).reshape(-1)

        # Albumentations가 image와 bbox를 같은 기하 변환으로 함께 갱신한다.
        #이미지는 넘파이로 받아야하고 박스는 리스트로 받아야한다.(albumen특징)
        transformed = self.pipeline(
            image=np.asarray(image),
            bboxes=boxes.tolist(),
            labels=labels.tolist(),
        )

        # image_id 같은 기존 target 정보는 유지하고, 변환된 boxes/labels만 교체한다.
        transformed_target = dict(target)
        transformed_target["boxes"] = torch.as_tensor(
            transformed["bboxes"],
            dtype=torch.float32,
        ).reshape(-1, 4)
        transformed_target["labels"] = torch.as_tensor(
            transformed["labels"],
            dtype=torch.int64,
        ).reshape(-1)

        # Dataset이 마지막 to_tensor를 담당하므로 여기서는 다시 PIL Image로 반환한다.
        return Image.fromarray(transformed["image"]), transformed_target


def train_transform(img_size: int) -> Transform:
    """학습용 transform 파이프라인을 반환한다.

    처음에는 val_transform과 동일하게 resize만 적용하고,
    이후 train 전용 증강은 이 함수에 추가한다.
    """
    # v0.1에서는 검증과 동일하게 resize만 적용하고, 이후 train 전용 증강을 여기에 추가한다.
    return Transform(_resize_pipeline(img_size))


def val_transform(img_size: int) -> Transform:
    """검증/테스트용 transform 파이프라인을 반환한다."""
    # 검증/테스트는 성능 측정을 흔들지 않도록 deterministic resize만 적용한다.
    return Transform(_resize_pipeline(img_size))


def _resize_pipeline(img_size: int) -> A.Compose:
    # pascal_voc은 [x_min, y_min, x_max, y_max] 형식으로, 프로젝트 내부 xyxy 계약과 같다.
    return A.Compose(
        [A.Resize(height=img_size, width=img_size)],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
        ),
    )
