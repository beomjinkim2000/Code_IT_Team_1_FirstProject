import torch


def test_cuda_availability_returns_bool():
    assert isinstance(torch.cuda.is_available(), bool)
