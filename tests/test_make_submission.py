"""make_submission.py 테스트

테스트 목적:
- postprocess.py 출력 형식인 Prediction list를 CSV row로 변환
- bbox가 xyxy에서 Kaggle 제출용 xywh로 변환되는지 확인
- annotation_id가 1부터 순서대로 증가하는지 확인
- 빈 prediction이 있어도 CSV 생성이 실패하지 않는지 확인
"""

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.submission.make_submission import (
    SUBMISSION_COLUMNS,
    make_submission,
    predictions_to_rows,
)


def make_mock_predictions() -> list[dict]:
    """submission 테스트에 사용할 가짜 postprocess 결과를 만든다."""

    return [
        {
            "image_id": 10,
            "boxes": torch.tensor(
                [
                    [10.0, 20.0, 50.0, 80.0],
                    [100.0, 120.0, 180.0, 220.0],
                ],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([3, 4], dtype=torch.int64),
            "scores": torch.tensor([0.9, 0.7], dtype=torch.float32),
        }
    ]


_MOCK_LABEL_TO_CATEGORY: dict[int, int] = {3: 3, 4: 4}


def test_predictions_to_rows_converts_xyxy_to_xywh() -> None:
    """Prediction list가 Kaggle 제출 row 형식으로 변환되는지 확인"""

    rows = predictions_to_rows(make_mock_predictions(), _MOCK_LABEL_TO_CATEGORY)

    assert len(rows) == 2

    assert rows[0]["annotation_id"] == 1
    assert rows[0]["image_id"] == 10
    assert rows[0]["category_id"] == 3
    assert rows[0]["bbox_x"] == 10.0
    assert rows[0]["bbox_y"] == 20.0
    assert rows[0]["bbox_w"] == 40.0
    assert rows[0]["bbox_h"] == 60.0
    assert rows[0]["score"] == pytest.approx(0.9)

    assert rows[1]["annotation_id"] == 2
    assert rows[1]["category_id"] == 4
    assert rows[1]["bbox_x"] == 100.0
    assert rows[1]["bbox_y"] == 120.0
    assert rows[1]["bbox_w"] == 80.0
    assert rows[1]["bbox_h"] == 100.0


def test_predictions_to_rows_keeps_custom_start_annotation_id() -> None:
    """annotation_id 시작 번호를 바꿀 수 있는지 확인"""

    rows = predictions_to_rows(
        make_mock_predictions(),
        _MOCK_LABEL_TO_CATEGORY,
        start_annotation_id=100,
    )

    assert rows[0]["annotation_id"] == 100
    assert rows[1]["annotation_id"] == 101


def test_make_submission_writes_csv(tmp_path: Path) -> None:
    """submission.csv 파일이 지정한 컬럼 순서로 저장되는지 확인"""

    output_path = tmp_path / "submission.csv"

    saved_path = make_submission(
        predictions=make_mock_predictions(),
        label_to_category=_MOCK_LABEL_TO_CATEGORY,
        output_path=output_path,
    )

    assert saved_path == output_path
    assert saved_path.exists()

    df = pd.read_csv(saved_path)

    assert list(df.columns) == SUBMISSION_COLUMNS
    assert len(df) == 2
    assert df.loc[0, "annotation_id"] == 1
    assert df.loc[0, "bbox_w"] == 40.0
    assert df.loc[0, "bbox_h"] == 60.0


def test_make_submission_allows_empty_predictions(tmp_path: Path) -> None:
    """예측 결과가 하나도 없어도 header만 있는 CSV를 생성하는지 확인"""

    output_path = tmp_path / "empty_submission.csv"

    saved_path = make_submission(
        predictions=[
            {
                "image_id": 99,
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64),
                "scores": torch.empty((0,), dtype=torch.float32),
            }
        ],
        label_to_category={},
        output_path=output_path,
    )

    df = pd.read_csv(saved_path)

    assert list(df.columns) == SUBMISSION_COLUMNS
    assert len(df) == 0