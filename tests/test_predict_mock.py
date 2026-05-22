"""MockDataset 기반 predict_batch 테스트.

MockDataset에서 이미지 batch를 만들고,
DummyModel로 가짜 model forward를 실행해
predict_batch()가 raw output을 반환하는지 확인한다.
"""




import torch
from torch import nn
from torch.utils.data import DataLoader

from src.engine.predict import predict_batch
from tests.mock_dataset import MockDataset

class DummyModel(nn.Module):
    """predict_batch 테스트용 모델"""

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = images.shape[0]

        return {
            "boxes": torch.zeros((batch_size, 1, 4), dtype=torch.float32),
            "labels": torch.ones((batch_size, 1), dtype=torch.int64),
            "scores": torch.ones((batch_size, 1), dtype=torch.float32),
        }
    # todo(issue #18): 실제 YOLO raw output 형식이 확정되면
    # 이 dummy output과 아래 assert를 실제 형식에 맞게 수정한다.


def collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[torch.Tensor, list[dict]]:
    """MockDataset 샘플을 batch tensor와 target list로 묶는다.

    todo: src/utils/collate.py가 구현되면 팀 공통 collate_fn으로 교체한다.
    """
    images, targets = zip(*batch)
    return torch.stack(list(images)), list(targets)



def test_predict_batch() -> None:
    """MockDataset으로 predict_batch 실행 흐름을 확인한다."""
    dataset = MockDataset(size=2, img_size=640, num_classes=10)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    images, targets = next(iter(dataloader))

    raw_predictions = predict_batch(
        model=DummyModel(),
        images=images,
        device="cpu",
    )

    assert images.shape == (2, 3, 640, 640)
    # 모듈 설계 기준: predict_batch 입력은 batch tensor [B, C, H, W]이다.

    assert len(targets) == 2
    # MockDataset target이 batch 크기만큼 유지되는지 확인한다.

    assert raw_predictions["boxes"].shape == (2, 1, 4)
    assert raw_predictions["labels"].shape == (2, 1)
    assert raw_predictions["scores"].shape == (2, 1)

    assert raw_predictions["boxes"].dtype == torch.float32
    assert raw_predictions["labels"].dtype == torch.int64
    assert raw_predictions["scores"].dtype == torch.float32
    # assert는 DummyModel 기준 임시값이다.
    # todo(issue #18): 실제 YOLO raw output 형식이 확정되면 assert를 교체.