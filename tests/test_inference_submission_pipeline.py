"""예측 → 후처리 → 제출 CSV 연결 테스트.

테스트 목적:
- predict_batch()가 raw output을 반환하는지 확인
- postprocess_raw_outputs()가 raw output을 Prediction list로 변환하는지 확인
- make_submission()이 Prediction list를 Kaggle CSV로 저장하는지 확인
- labels(class_id)가 category_id로 매핑되는지 확인
"""

from pathlib import Path

import pandas as pd
import torch

from src.engine.postprocess import PostprocessConfig, postprocess_raw_outputs
from src.engine.predict import predict_batch
from src.submission.make_submission import SUBMISSION_COLUMNS, make_submission
from tests.dummy_model import DummyModel


def test_predict_postprocess_submission_pipeline(tmp_path: Path) -> None:
    """DummyModel 기준 전체 예측 제출 파이프라인 포맷을 확인한다."""

    num_classes = 10
    batch_size = 2
    img_size = 640
    image_ids = [100, 101]

    model = DummyModel(num_classes=num_classes)
    images = torch.rand(batch_size, 3, img_size, img_size)

    raw_outputs = predict_batch(
        model=model,
        images=images,
        device="cpu",
    )

    predictions = postprocess_raw_outputs(
        raw_outputs=raw_outputs,
        image_ids=image_ids,
        config=PostprocessConfig(
            conf_threshold=0.99,
            iou_threshold=0.7,
            max_detections=4,
        ),
    )

    assert len(predictions) == batch_size

    for prediction, image_id in zip(predictions, image_ids):
        assert prediction["image_id"] == image_id
        assert prediction["boxes"].shape[1] == 4
        assert prediction["labels"].ndim == 1
        assert prediction["scores"].ndim == 1
        assert prediction["boxes"].shape[0] <= 4
        assert prediction["labels"].shape[0] <= 4
        assert prediction["scores"].shape[0] <= 4
        assert prediction["boxes"].dtype == torch.float32
        assert prediction["labels"].dtype == torch.int64
        assert prediction["scores"].dtype == torch.float32

    label_to_category = {
        class_id: 1000 + class_id
        for class_id in range(num_classes)
    }

    output_path = tmp_path / "submission.csv"

    saved_path = make_submission(
        predictions=predictions,
        label_to_category=label_to_category,
        output_path=output_path,
    )

    assert saved_path == output_path
    assert saved_path.exists()

    df = pd.read_csv(saved_path)

    assert list(df.columns) == SUBMISSION_COLUMNS
    assert len(df) <= batch_size * 4

    if len(df) > 0:
        assert set(df["image_id"]).issubset(set(image_ids))
        assert df["category_id"].between(1000, 1000 + num_classes - 1).all()
        assert (df["bbox_w"] >= 0).all()
        assert (df["bbox_h"] >= 0).all()
        assert df["score"].between(0.0, 1.0).all() 