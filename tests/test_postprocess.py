"""postprocess.py 테스트 파일

테스트 목적:
- score threshold보다 낮은 예측 제거
- score 기준 내림차순 정렬
- max_detections 개수 제한
- 빈 예측 결과의 shape / dtype 유지
- 프로젝트 공용 dummy prediction fixture와의 호환성 확인
"""

import torch

from src.engine.postprocess import (
  PostprocessConfig,
  postprocess_prediction,
  postprocess_predictions,
  postprocess_raw_outputs,
)
from tests.dummy_model import make_dummy_pred_dict, make_dummy_raw_output

def make_mock_prediction() -> dict:
 """후처리 규칙 명확히 롹인하기 위한 가짜 예측 결과를 생성
  scores를 [0.9, 0.2, 0.7]로 고정해서 threshold filtering과
    score 정렬이 제대로 동작하는지 확인한다.
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
    """이미지 1장 prediction의 필터링 / 정렬 / 개수 제한을 확인."""

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

    # score 기준 내림차순 정렬 확인
    assert processed["scores"][0] >= processed["scores"][1]

    # score 0.2 예측은 threshold보다 낮아서 제거되어야 한다.
    assert torch.all(processed["scores"] >= 0.25)

    # interfaces.md 형식 확인
    assert processed["boxes"].dtype == torch.float32
    assert processed["labels"].dtype == torch.int64
    assert processed["scores"].dtype == torch.float32

def test_postprocess_predictions() -> None:
    """여러 이미지 prediction을 list 단위로 후처리하는 helper를 확인."""

    config = PostprocessConfig(conf_threshold=0.25, max_detections=2)

    processed_list = postprocess_predictions([make_mock_prediction()], config)

    assert len(processed_list) == 1
    assert processed_list[0]["image_id"] == 0
    assert processed_list[0]["boxes"].shape == (2, 4)

def test_postprocess_empty_prediction() -> None:
    """threshold를 통과한 예측이 없을 때도 빈 prediction 형식이 유지되는지 확인"""

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

def test_postprocess_with_project_dummy_prediction() -> None:
    """공용 dummy prediction fixture도 postprocess.py 입력 형식과 맞는지 확인한다."""

    predictions = make_dummy_pred_dict(
        batch_size=1,
        num_classes=10,
        num_detections=3,
        start_image_id=2,
    )
    prediction = predictions[0]

    processed = postprocess_prediction(
        prediction,
        PostprocessConfig(conf_threshold=0.0, max_detections=4),
    )

    assert processed["image_id"] == 2
    assert processed["boxes"].shape[1] == 4
    assert processed["labels"].ndim == 1
    assert processed["scores"].ndim == 1
    assert processed["boxes"].dtype == torch.float32
    assert processed["labels"].dtype == torch.int64
    assert processed["scores"].dtype == torch.float32

def test_postprocess_raw_outputs_from_dummy_model_format() -> None:
    """DummyModel raw output을 Prediction 리스트로 변환할 수 있는지 확인"""

    raw_outputs = (
        make_dummy_raw_output(
            batch_size=2,
            num_classes=10,
        ),
    )

    processed_list = postprocess_raw_outputs(
        raw_outputs=raw_outputs,
        image_ids=[10, 11],
        config=PostprocessConfig(
            conf_threshold=0.99,
            iou_threshold=0.7,
            max_detections=4,
        ),
    )

    assert len(processed_list) == 2
    assert processed_list[0]["image_id"] == 10
    assert processed_list[1]["image_id"] == 11

    for processed in processed_list:
        assert processed["boxes"].shape[1] == 4
        assert processed["labels"].ndim == 1
        assert processed["scores"].ndim == 1
        assert processed["boxes"].shape[0] <= 4
        assert processed["labels"].shape[0] <= 4
        assert processed["scores"].shape[0] <= 4
        assert processed["boxes"].dtype == torch.float32
        assert processed["labels"].dtype == torch.int64
        assert processed["scores"].dtype == torch.float32