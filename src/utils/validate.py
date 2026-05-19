from typing import List, Dict
import torch
from torch import Tensor


def validate_batch(images: List[Tensor], targets: List[Dict], img_size: int) -> None:
    """
    DataLoader에서 꺼낸 배치가 interfaces.md 스펙을 만족하는지 확인.
    조건 불만족 시 AssertionError — 학습 루프 진입 전에 호출할 것.
    """
    assert len(images) == len(targets), (
        f"images({len(images)})와 targets({len(targets)}) 개수가 다름"
    )

    for i, (image, target) in enumerate(zip(images, targets)):
        # --- image 검증 ---
        assert isinstance(image, Tensor), \
            f"[{i}] image가 Tensor가 아님: {type(image)}"
        assert image.ndim == 3, \
            f"[{i}] image shape이 [C,H,W]가 아님: {image.shape}"
        assert image.shape[0] == 3, \
            f"[{i}] 채널이 3이 아님: {image.shape}"
        assert image.shape[1] == img_size and image.shape[2] == img_size, \
            f"[{i}] 이미지 크기가 {img_size}x{img_size}가 아님: {image.shape[1:]}"
        assert image.dtype == torch.float32, \
            f"[{i}] image dtype이 float32가 아님: {image.dtype}"
        assert image.min() >= 0.0 and image.max() <= 1.0, \
            f"[{i}] image 값 범위가 0~1이 아님: min={image.min():.3f}, max={image.max():.3f}"

        # --- target 키 검증 ---
        for key in ("boxes", "labels", "image_id"):
            assert key in target, f"[{i}] target에 '{key}' 키 없음"

        boxes  = target["boxes"]
        labels = target["labels"]

        # --- boxes 검증 ---
        assert isinstance(boxes, Tensor), \
            f"[{i}] boxes가 Tensor가 아님: {type(boxes)}"
        assert boxes.ndim == 2 and boxes.shape[1] == 4, \
            f"[{i}] boxes shape이 [N,4]가 아님: {boxes.shape}"
        assert boxes.dtype == torch.float32, \
            f"[{i}] boxes dtype이 float32가 아님: {boxes.dtype}"

        if boxes.shape[0] > 0:
            assert (boxes[:, 2] > boxes[:, 0]).all(), \
                f"[{i}] x2 > x1 조건 불만족 (xyxy 형식 확인)"
            assert (boxes[:, 3] > boxes[:, 1]).all(), \
                f"[{i}] y2 > y1 조건 불만족 (xyxy 형식 확인)"

        # --- labels 검증 ---
        assert isinstance(labels, Tensor), \
            f"[{i}] labels가 Tensor가 아님: {type(labels)}"
        assert labels.ndim == 1, \
            f"[{i}] labels shape이 [N]이 아님: {labels.shape}"
        assert labels.dtype == torch.int64, \
            f"[{i}] labels dtype이 int64가 아님: {labels.dtype}"
        assert labels.shape[0] == boxes.shape[0], \
            f"[{i}] boxes({boxes.shape[0]})와 labels({labels.shape[0]}) 개수 불일치"
