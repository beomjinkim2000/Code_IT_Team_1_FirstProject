import torch
from tests.mock_dataset import MockDataset
from src.models.baseline import build_model


def test_build_model_output_type():
    model = build_model(num_classes=10)
    ds = MockDataset(size=2, num_classes=10)
    images = torch.stack([ds[i][0] for i in range(2)])
    out = model(images)
    assert isinstance(out, tuple)


def test_build_model_num_classes():
    model = build_model(num_classes=10)
    assert model.nc == 10


def test_build_model_output_channels():
    num_classes = 10
    model = build_model(num_classes=num_classes)
    detect_head = model.model[-1]
    assert detect_head.nc == num_classes
    assert detect_head.no == num_classes + detect_head.reg_max * 4
