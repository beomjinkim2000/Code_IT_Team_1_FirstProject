"""score 0.2 예측 제거
score 높은 순서로 정렬
max_detections=2 적용
boxes / labels / scores shape 유지
dtype 계약 유지
여러 이미지용 postprocess_predictions()도 작동
"""

import torch

from src.engine.postprocess import (
    PostprocessConfig,
    postprocess_prediction,
    postprocess_predictions,
)

def make_mock_prediction() -> dict:
    """postprocess.py 테스트에 사용할 가짜 예측 결과를 만든다.

    실제 모델 예측처럼 boxes, labels, scores를 가진 dict를 반환한다.
    scores를 일부러 [0.9, 0.2, 0.7]로 두어 confidence filtering과
    score 정렬이 제대로 동작하는지 확인할 수 있게 한다.
    """
    return {
        "image_id": 0,
        "boxes": torch.tensor(
            [
                [10.0, 20.0, 50.0, 80.0],
                [15.0, 25.0, 55.0, 85.0],
                [100.0, 120.0, 180.0, 220.0],
            ],
            dtype=torch.float32,
        ),
        "labels": torch.tensor([1, 1, 2], dtype=torch.int64),
        "scores": torch.tensor([0.9, 0.2, 0.7], dtype=torch.float32),
    }

def test_postprocess_prediction() -> None:
    """Mock prediction 1개를 후처리해서 필터링/정렬/개수 제한을 확인."""
    config = PostprocessConfig(
        conf_threshold=0.25,
        iou_threshold=0.7,
        max_detections=2,
    )

    processed = postprocess_prediction(make_mock_prediction(), config)

    assert processed["image_id"] == 0

    assert processed["boxes"].shape == (2, 4)
    assert processed["labels"].shape == (2,)
    assert processed["scores"].shape == (2,)
    # score 0.2는 threshold보다 낮아서 제거
    # max_detections=2이므로 최종 예측은 2개만 남아야 한다.

    assert processed["scores"][0] >= processed["scores"][1]
    # score 기준 내림차순 정렬 확인

    assert processed["boxes"].dtype == torch.float32
    assert processed["labels"].dtype == torch.int64
    assert processed["scores"].dtype == torch.float32
    # interfaces.md 형식 확인

def test_postprocess_predictions() -> None:
    """여러 이미지 예측을 list 단위로 후처리하는 helper를 확인한다."""
    config = PostprocessConfig(conf_threshold=0.25, max_detections=2)

    processed_list = postprocess_predictions([make_mock_prediction()], config)

    assert len(processed_list) == 1
    assert processed_list[0]["image_id"] == 0
    assert processed_list[0]["boxes"].shape == (2, 4)

def test_postprocess_empty_prediction() -> None:
    """threshold 통과 예측이 없을 때 빈 prediction 형식이 유지되는지 확인."""
    prediction = {
        "image_id": 1,
        "boxes": torch.tensor([[10.0, 20.0, 50.0, 80.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "scores": torch.tensor([0.1], dtype=torch.float32),
    }

    processed = postprocess_prediction(
        prediction,
        PostprocessConfig(conf_threshold=0.25),
    )

    assert processed["image_id"] == 1
    assert processed["boxes"].shape == (0, 4)
    assert processed["labels"].shape == (0,)
    assert processed["scores"].shape == (0,)
    assert processed["boxes"].dtype == torch.float32
    assert processed["labels"].dtype == torch.int64
    assert processed["scores"].dtype == torch.float32
