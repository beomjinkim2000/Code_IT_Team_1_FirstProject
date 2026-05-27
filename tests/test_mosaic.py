import torch
import pytest
from torch.utils.data import Dataset

from src.data.mosaic import MosaicDataset


class MockDataset(Dataset):
    def __init__(self, size: int = 4, img_size: int = 64, n_boxes: int = 1) -> None:
        self.size = size
        self.img_size = img_size
        self.n_boxes = n_boxes

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        img = torch.rand(3, self.img_size, self.img_size)
        if self.n_boxes > 0:
            boxes = torch.tensor([[10.0, 10.0, 30.0, 30.0]] * self.n_boxes)
            labels = torch.zeros(self.n_boxes, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": idx + 1,
            "original_size": (self.img_size, self.img_size),
        }
        return img, target


def test_mosaic_dataset_len() -> None:
    inner = MockDataset(size=8, img_size=64)
    ds = MosaicDataset(inner, img_size=64, p=1.0)
    assert len(ds) == len(inner)


def test_mosaic_output_interface_contract() -> None:
    img_size = 64
    inner = MockDataset(size=4, img_size=img_size)
    ds = MosaicDataset(inner, img_size=img_size, p=1.0)
    img, target = ds[0]

    assert img.shape == (3, img_size, img_size)
    assert img.dtype == torch.float32

    boxes = target["boxes"]
    labels = target["labels"]
    assert isinstance(boxes, torch.Tensor)
    assert boxes.ndim == 2
    assert boxes.shape[1] == 4
    assert boxes.dtype == torch.float32
    assert isinstance(labels, torch.Tensor)
    assert labels.dtype == torch.int64
    assert labels.shape[0] == boxes.shape[0]
    assert isinstance(target["image_id"], int)


def test_mosaic_boxes_within_bounds() -> None:
    img_size = 64
    inner = MockDataset(size=4, img_size=img_size)
    ds = MosaicDataset(inner, img_size=img_size, p=1.0)
    _, target = ds[0]
    boxes = target["boxes"]
    if boxes.numel() > 0:
        assert torch.all(boxes >= 0)
        assert torch.all(boxes <= img_size)


def test_mosaic_p0_returns_inner_dataset_item() -> None:
    img_size = 64
    inner = MockDataset(size=4, img_size=img_size)
    ds = MosaicDataset(inner, img_size=img_size, p=0.0)

    torch.manual_seed(0)
    img_inner, _ = inner[0]

    torch.manual_seed(0)
    img_mosaic, _ = ds[0]

    assert torch.allclose(img_inner, img_mosaic)


def test_mosaic_empty_boxes_handled() -> None:
    img_size = 64
    inner = MockDataset(size=4, img_size=img_size, n_boxes=0)
    ds = MosaicDataset(inner, img_size=img_size, p=1.0)
    _, target = ds[0]
    boxes = target["boxes"]
    labels = target["labels"]
    assert boxes.shape == (0, 4)
    assert labels.shape == (0,)
    assert boxes.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_mosaic_min_bbox_size_filters_small_boxes() -> None:
    img_size = 64

    class TinyBoxDataset(Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, idx: int):
            img = torch.rand(3, img_size, img_size)
            # 원본 2×2 박스 → 0.5 배율 후 1×1 → min_bbox_size=2 이므로 제거 대상
            boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
            labels = torch.zeros(1, dtype=torch.int64)
            return img, {
                "boxes": boxes,
                "labels": labels,
                "image_id": idx + 1,
                "original_size": (img_size, img_size),
            }

    ds = MosaicDataset(TinyBoxDataset(), img_size=img_size, p=1.0, min_bbox_size=2)
    _, target = ds[0]
    assert target["boxes"].shape[0] == 0


def test_mosaic_labels_match_boxes_count() -> None:
    img_size = 64
    inner = MockDataset(size=4, img_size=img_size, n_boxes=2)
    ds = MosaicDataset(inner, img_size=img_size, p=1.0, min_bbox_size=1)
    _, target = ds[0]
    assert target["boxes"].shape[0] == target["labels"].shape[0]
