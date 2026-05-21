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

def main() -> None:
    """Mock prediction을 postprocess.py에 넣어 1회 실행 흐름을 확인.

    확인하는 내용:
    - confidence threshold보다 낮은 예측이 제거되는지
    - score 기준 내림차순 정렬이 되는지
    - max_detections 제한이 적용되는지
    - interfaces.md 형식을 지키는지
    """
    config = PostprocessConfig(
        conf_threshold=0.25,
        iou_threshold=0.7,
        max_detections=2,
    )

    prediction = make_mock_prediction()
    processed = postprocess_prediction(prediction, config)

    assert processed["image_id"] == 0

    assert processed["boxes"].shape == (2, 4)
    assert processed["labels"].shape == (2,)
    assert processed["scores"].shape == (2,) 
    # max_detections=2이므로 최종 예측은 최대 2개만 남아야 한다.
    # score 0.2인 예측은 conf_threshold=0.25보다 낮으므로 제거되어야 한다.

    assert processed["scores"][0] >= processed["scores"][1]
    # 후처리 결과는 score 기준 내림차순이어야 한다.

    assert processed["boxes"].dtype == torch.float32
    assert processed["labels"].dtype == torch.int64
    assert processed["scores"].dtype == torch.float32
    # interfaces.md 형식의 dtype을 지켜야 한다.

    processed_list = postprocess_predictions([prediction], config)
    # 여러 이미지 예측을 한 번에 처리하는 helper도 같은 기준으로 동작해야 한다.

    assert len(processed_list) == 1
    assert processed_list[0]["boxes"].shape == (2, 4)

    print("postprocess mock 1-pass test passed")


if __name__ == "__main__":
    main()