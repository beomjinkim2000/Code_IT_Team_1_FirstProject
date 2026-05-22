import torch

from src.data.dataset import PillDataset, RAW_DATA_ROOT
from src.data.transforms import val_transform
from src.utils.config import load_config
from src.utils.validate import validate_dataset


def _sample_image_files(count: int = 2) -> list[str]:
    image_dir = RAW_DATA_ROOT / "train_images"
    return [path.name for path in sorted(image_dir.glob("*.png"))[:count]]


def _build_dataset() -> PillDataset:
    cfg = load_config()
    img_size = cfg["train"]["img_size"]
    return PillDataset(
        transforms=val_transform(img_size),
        image_files=_sample_image_files(),
        category_to_label=cfg["data"]["category_to_label"],
    )


def test_pill_dataset_passes_common_validate_contract() -> None:
    cfg = load_config()
    dataset = _build_dataset()

    validate_dataset(dataset, img_size=cfg["train"]["img_size"], sample_size=len(dataset))


def test_pill_dataset_returns_expected_target_contract() -> None:
    cfg = load_config()
    img_size = cfg["train"]["img_size"]
    dataset = _build_dataset()
    image, target = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, img_size, img_size)
    assert image.dtype == torch.float32
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0

    assert set(target) == {"boxes", "labels", "image_id"}

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


def test_pill_dataset_image_files_filter_limits_dataset() -> None:
    image_files = _sample_image_files(count=1)
    cfg = load_config()

    dataset = PillDataset(
        transforms=val_transform(cfg["train"]["img_size"]),
        image_files=image_files,
        category_to_label=cfg["data"]["category_to_label"],
    )

    assert len(dataset) == 1
    assert dataset.image_paths[0].name == image_files[0]
