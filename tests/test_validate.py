import torch
import pytest
from tests.mock_dataset import MockDataset
from src.utils.validate import validate_dataset, validate_batch


# ── validate_dataset ─────────────────────────────────────────────────────────

def test_valid_passes():
    validate_dataset(MockDataset(size=2), 640)


@pytest.mark.parametrize("rule", ["not_tensor", "wrong_ndim", "wrong_channel", "wrong_size"])
def test_hard_rules_raise(rule):
    with pytest.raises(AssertionError):
        validate_dataset(MockDataset(size=2, break_rule=rule), 640)


@pytest.mark.parametrize("rule", ["wrong_dtype", "wrong_range"])
def test_soft_rules_pass(rule):
    validate_dataset(MockDataset(size=2, break_rule=rule), 640)


def test_img_size_mismatch_raises():
    with pytest.raises(AssertionError):
        validate_dataset(MockDataset(size=2, img_size=320), 640)


def test_empty_dataset_raises():
    with pytest.raises(AssertionError):
        validate_dataset(MockDataset(size=0), 640)


# ── validate_batch ───────────────────────────────────────────────────────────

def test_validate_batch_valid():
    ds = MockDataset(size=2)
    images  = [ds[i][0] for i in range(2)]
    targets = [ds[i][1] for i in range(2)]
    validate_batch(images, targets, 640)


def test_validate_batch_length_mismatch():
    ds = MockDataset(size=2)
    with pytest.raises(AssertionError):
        validate_batch([ds[0][0]], [ds[0][1], ds[1][1]], 640)


def test_validate_batch_missing_key():
    ds = MockDataset(size=1)
    target = {"boxes": ds[0][1]["boxes"]}  # labels, image_id 없음
    with pytest.raises(AssertionError):
        validate_batch([ds[0][0]], [target], 640)


def test_validate_batch_invalid_bbox():
    ds = MockDataset(size=1)
    target = {
        "boxes":    torch.tensor([[100.0, 100.0, 50.0, 150.0]]),  # x2 < x1
        "labels":   torch.tensor([0], dtype=torch.int64),
        "image_id": 0,
    }
    with pytest.raises(AssertionError):
        validate_batch([ds[0][0]], [target], 640)
