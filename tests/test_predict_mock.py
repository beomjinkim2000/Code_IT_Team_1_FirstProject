"""MockDataset 기반 predict_batch 테스트.

MockDataset에서 이미지 batch를 만들고,
DummyModel로 가짜 model forward를 실행해
predict_batch()가 raw output을 반환하는지 확인한다.
"""




import torch
from torch.utils.data import DataLoader

from src.engine.predict import predict_batch
from src.utils.collate import collate_fn
from tests.dummy_model import NUM_ANCHORS, DummyModel
from tests.mock_dataset import MockDataset



def test_predict_batch() -> None:
    """MockDataset으로 predict_batch 실행 흐름을 확인한다."""
    dataset = MockDataset(size=2, img_size=640, num_classes=10)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    image_list, targets = next(iter(dataloader))
    images = torch.stack(image_list)

    raw_predictions = predict_batch(
        model=DummyModel(num_classes=10),
        images=images,
        device="cpu",
    )

    assert images.shape == (2, 3, 640, 640)
    # 모듈 설계 기준: predict_batch 입력은 batch tensor [B, C, H, W]이다.

    assert len(targets) == 2
    # MockDataset target이 batch 크기만큼 유지되는지 확인한다.

    assert isinstance(raw_predictions, tuple)
    assert len(raw_predictions) == 1

    raw_output = raw_predictions[0]
    assert raw_output.shape == (2, 4 + 10, NUM_ANCHORS)
    assert raw_output.dtype == torch.float32
    # #43 DummyModel 기준 실제 YOLO raw output 형식을 흉내낸 값이다.
