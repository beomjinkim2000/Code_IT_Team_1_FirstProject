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


def train_transform(img_size: int, aug_cfg: dict | None = None) -> Transform:
    """학습용 transform 파이프라인을 반환한다."""
    c = aug_cfg or {}
    tile = c["clahe_tile_grid_size"]
    rotate_limit = c["safe_rotate_limit"]
    pipeline = A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=c["horizontal_flip_p"]),
            A.VerticalFlip(p=c["vertical_flip_p"]),
            A.RandomRotate90(p=c["random_rotate90_p"]),
            A.SafeRotate(limit=(-rotate_limit, rotate_limit), p=c["safe_rotate_p"]),
            A.RandomBrightnessContrast(brightness_limit=c["brightness_limit"], contrast_limit=c["contrast_limit"], p=c["brightness_contrast_p"]),
            A.HueSaturationValue(hue_shift_limit=c["hue_shift_limit"], sat_shift_limit=c["sat_shift_limit"], val_shift_limit=c["val_shift_limit"], p=c["hue_saturation_p"]),
            A.CLAHE(clip_limit=c["clahe_clip_limit"], tile_grid_size=(tile, tile), p=c["clahe_p"]),
            A.Sharpen(alpha=(c["sharpen_alpha_min"], c["sharpen_alpha_max"]), lightness=(c["sharpen_lightness_min"], c["sharpen_lightness_max"]), p=c["sharpen_p"]),
            A.GaussNoise(var_limit=(c["gauss_noise_var_min"], c["gauss_noise_var_max"]), p=c["gauss_noise_p"]),
            A.RandomScale(scale_limit=c["random_scale_limit"], p=c["random_scale_p"]),
            A.Resize(height=img_size, width=img_size),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_area=c["min_area"],
            min_visibility=c["min_visibility"],
        ),
    )
    return Transform(pipeline)


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
