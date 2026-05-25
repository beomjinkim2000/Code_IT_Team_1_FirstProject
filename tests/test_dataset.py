import json
from pathlib import Path

import torch
from PIL import Image

from src.data.dataset import PillDataset
from src.data.transforms import val_transform
from src.utils.config import load_config
from src.utils.validate import validate_dataset


def _make_raw_dataset(root: Path, category_id: int, image_count: int = 2) -> list[str]:
    image_dir = root / "train_images"
    annotation_dir = root / "train_annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)

    image_files: list[str] = []
    images: list[dict] = []
    annotations: list[dict] = []

    for idx in range(image_count):
        image_id = idx + 1
        file_name = f"sample_{image_id}.png"
        image_files.append(file_name)
        images.append({"id": image_id, "file_name": file_name})

        # 원본 이미지는 정사각형이 아니어도 transforms에서 img_size x img_size로 맞춘다.
        Image.new("RGB", (976, 1280), color=(64 + idx, 96, 128)).save(image_dir / file_name)
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [100.0, 200.0, 120.0, 160.0],
            }
        )

    with open(annotation_dir / "annotations.json", "w", encoding="utf-8") as f:
        json.dump({"images": images, "annotations": annotations}, f)

    return image_files


def _build_dataset(root: Path) -> PillDataset:
    cfg = load_config()
    img_size = cfg["train"]["img_size"]
    category_id = next(iter(cfg["data"]["category_to_label"]))
    image_files = _make_raw_dataset(root, category_id=category_id)

    return PillDataset(
        root=root,
        transforms=val_transform(img_size),
        image_files=image_files,
        category_to_label=cfg["data"]["category_to_label"],
    )


def test_pill_dataset_passes_common_validate_contract(tmp_path: Path) -> None:
    cfg = load_config()
    dataset = _build_dataset(tmp_path)

    validate_dataset(dataset, img_size=cfg["train"]["img_size"], sample_size=len(dataset))


def test_pill_dataset_returns_expected_target_contract(tmp_path: Path) -> None:
    cfg = load_config()
    img_size = cfg["train"]["img_size"]
    dataset = _build_dataset(tmp_path)
    image, target = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, img_size, img_size)
    assert image.dtype == torch.float32
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0

    assert set(target) == {"boxes", "labels", "image_id", "original_size"}

    boxes = target["boxes"]
    labels = target["labels"]
    image_id = target["image_id"]

    assert isinstance(boxes, torch.Tensor)
    assert boxes.ndim == 2
    assert boxes.shape[1] == 4
    assert boxes.dtype == torch.float32

    assert isinstance(labels, torch.Tensor)
    assert labels.ndim == 1
    assert labels.dtype == torch.int64
    assert labels.shape[0] == boxes.shape[0]

    assert isinstance(image_id, int)

    if boxes.shape[0] > 0:
        assert torch.all(boxes[:, 2] > boxes[:, 0])
        assert torch.all(boxes[:, 3] > boxes[:, 1])
        assert torch.all(boxes >= 0)
        assert torch.all(boxes <= img_size)
        assert int(labels.min()) >= 0
        assert int(labels.max()) < 56


def test_pill_dataset_image_files_filter_limits_dataset(tmp_path: Path) -> None:
    cfg = load_config()
    category_id = next(iter(cfg["data"]["category_to_label"]))
    image_files = _make_raw_dataset(tmp_path, category_id=category_id, image_count=2)

    dataset = PillDataset(
        root=tmp_path,
        transforms=val_transform(cfg["train"]["img_size"]),
        image_files=image_files[:1],
        category_to_label=cfg["data"]["category_to_label"],
    )

    assert len(dataset) == 1
    assert dataset.image_paths[0].name == image_files[0]
