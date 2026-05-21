"""MockDataset에서 이미지 2장 가져오기
DataLoader로 batch 만들기
DummyModel로 가짜 model forward 실행
predict_batch()가 raw prediction 반환하는지 확인
predict_images()가 interfaces.md 형식의 dummy prediction 반환하는지 확인
"""




import torch
from torch import nn
from torch.utils.data import DataLoader

from src.engine.predict import predict_batch, predict_images
from tests.mock_dataset import MockDataset

class DummyModel(nn.Module):
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = images.shape[0]

        return {
            "boxes": torch.tensor(
                [[[10.0, 20.0, 50.0, 80.0]]] * batch_size,
                dtype=torch.float32,
            ),
            "labels": torch.tensor([[1]] * batch_size, dtype=torch.int64),
            "scores": torch.tensor([[0.9]] * batch_size, dtype=torch.float32),
        }


def collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[torch.Tensor, list[dict]]:
    images, targets = zip(*batch)
    return torch.stack(list(images)), list(targets)


def main() -> None:
    dataset = MockDataset(size=2, img_size=640, num_classes=10)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    images, targets = next(iter(dataloader))

    model = DummyModel()
    raw_predictions = predict_batch(
        model=model,
        images=images,
        device="cpu",
    )

    assert images.shape == (2, 3, 640, 640)
    assert len(targets) == 2
    assert raw_predictions["boxes"].shape == (2, 1, 4)
    assert raw_predictions["labels"].shape == (2, 1)
    assert raw_predictions["scores"].shape == (2, 1)

    preds = predict_images(model=None, image_paths=["a.jpg", "b.jpg"])

    assert len(preds) == 2
    assert preds[0]["image_id"] == 0
    assert preds[0]["boxes"].shape == (1, 4)
    assert preds[0]["labels"].dtype == torch.int64
    assert preds[0]["scores"].dtype == torch.float32

    print("predict mock dataset 1-pass test passed")


if __name__ == "__main__":
    main()