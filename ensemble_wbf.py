"""WBF (Weighted Boxes Fusion) 앙상블.

여러 체크포인트의 predictions.json을 합쳐 단일 submission.csv를 생성한다.
기존 predictions.json을 그대로 사용하며, 이미지 크기는 테스트 이미지 폴더에서 직접 읽는다.

사용법:
    python ensemble_wbf.py \
        --preds outputs/predictions/pred_a.json outputs/predictions/pred_b.json \
        --weights 2 1 \
        --iou-thr 0.5 \
        --skip-box-thr 0.25 \
        --output outputs/submissions/submission_wbf.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ensemble_boxes import weighted_boxes_fusion

from src.data.dataset import RAW_DATA_ROOT
from src.submission.make_submission import make_submission
from src.utils.config import load_config


def _load_preds(path: str) -> dict[int, dict]:
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return {item["image_id"]: item for item in items}


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """정규화 좌표 [x1, y1, x2, y2] 두 박스의 IoU."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _soft_vote_labels(
    fused_boxes: np.ndarray,
    boxes_list: list[np.ndarray],
    class_probs_list: list[np.ndarray],
    weights: list[float],
    iou_thr: float,
) -> np.ndarray:
    """각 fused box에 대해 겹치는 원본 박스들의 class_probs를 가중 평균 후 argmax.

    class_probs가 없는 모델(기존 predictions.json)은 건너뜀.
    모든 모델에 class_probs가 없으면 None 반환 → WBF 기본 label 사용.
    """
    if all(cp is None for cp in class_probs_list):
        return None

    result = []
    for fbox in fused_boxes:
        weighted_sum = None
        total_w = 0.0
        for boxes, class_probs, w in zip(boxes_list, class_probs_list, weights):
            if class_probs is None or len(boxes) == 0:
                continue
            for box, probs in zip(boxes, class_probs):
                if _box_iou(fbox, box) >= iou_thr:
                    if weighted_sum is None:
                        weighted_sum = np.zeros(len(probs), dtype=np.float64)
                    weighted_sum += w * np.array(probs, dtype=np.float64)
                    total_w += w
        if weighted_sum is not None and total_w > 0:
            result.append(int(np.argmax(weighted_sum / total_w)))
        else:
            result.append(-1)  # fallback: WBF 기본값 사용
    return np.array(result)


def _build_image_size_map(test_image_dir: Path) -> dict[int, tuple[int, int]]:
    """image_id → (orig_w, orig_h) 맵을 테스트 이미지 파일에서 만든다."""
    size_map = {}
    for img_path in test_image_dir.glob("*.png"):
        stem = img_path.stem
        image_id = int(stem) if stem.isdigit() else None
        if image_id is None:
            continue
        with Image.open(img_path) as img:
            size_map[image_id] = img.size  # (width, height)
    return size_map


def _to_wbf_input(item: dict, orig_w: int, orig_h: int):
    boxes = np.array(item["boxes"], dtype=np.float32)
    scores = np.array(item["scores"], dtype=np.float32)
    labels = np.array(item["labels"], dtype=np.float32)

    if len(boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    # WBF 입력은 [0, 1] 정규화 좌표
    boxes[:, [0, 2]] /= orig_w
    boxes[:, [1, 3]] /= orig_h
    boxes = boxes.clip(0.0, 1.0)
    return boxes, scores, labels


def main():
    parser = argparse.ArgumentParser(description="WBF 앙상블 — predictions.json 여러 개를 합쳐 submission 생성")
    parser.add_argument(
        "--preds", nargs="+", required=True,
        help="합칠 predictions.json 경로 목록 (2개 이상)",
    )
    parser.add_argument(
        "--weights", nargs="+", type=float, default=None,
        help="모델별 가중치 (생략 시 동일 가중치). 성능 좋은 모델에 높게.",
    )
    parser.add_argument("--iou-thr", type=float, default=0.5, help="WBF IoU 임계값 (기본 0.5)")
    parser.add_argument("--skip-box-thr", type=float, default=0.25, help="이 score 미만 박스 무시 (기본 0.25)")
    parser.add_argument("--max-det", type=int, default=4, help="이미지당 최대 탐지 수 (기본 4)")
    parser.add_argument(
        "--output", default="outputs/submissions/submission_wbf.csv",
        help="결과 submission.csv 저장 경로",
    )
    args = parser.parse_args()

    if len(args.preds) < 2:
        raise ValueError("--preds에 predictions.json을 2개 이상 전달해야 합니다.")

    weights = args.weights or [1.0] * len(args.preds)
    if len(weights) != len(args.preds):
        raise ValueError("--weights 개수가 --preds 개수와 달라요.")

    cfg = load_config()
    label_to_category = cfg["data"]["label_to_category"]

    test_image_dir = RAW_DATA_ROOT / "test_images"
    print(f"테스트 이미지 크기 로드: {test_image_dir}")
    size_map = _build_image_size_map(test_image_dir)

    print(f"로드: {args.preds}")
    all_pred_dicts = [_load_preds(p) for p in args.preds]

    image_ids = sorted(all_pred_dicts[0].keys())
    print(f"이미지 수: {len(image_ids)}, 모델 수: {len(args.preds)}, 가중치: {weights}")

    final_predictions = []

    for image_id in image_ids:
        orig_w, orig_h = size_map[image_id]

        boxes_list, scores_list, labels_list, class_probs_list = [], [], [], []
        for pred_dict in all_pred_dicts:
            item = pred_dict.get(image_id, {"boxes": [], "scores": [], "labels": []})
            b, s, l = _to_wbf_input(item, orig_w, orig_h)
            boxes_list.append(b)
            scores_list.append(s)
            labels_list.append(l)
            cp_raw = item.get("class_probs")
            class_probs_list.append(np.array(cp_raw, dtype=np.float32) if cp_raw else None)

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=args.iou_thr,
            skip_box_thr=args.skip_box_thr,
        )

        # score 내림차순이므로 앞에서 max_det개만 자름
        boxes = boxes[: args.max_det]
        scores = scores[: args.max_det]
        labels = labels[: args.max_det]

        # 클래스 소프트 보팅 — class_probs가 있는 모델이 하나라도 있으면 적용
        if len(boxes) > 0:
            voted = _soft_vote_labels(boxes, boxes_list, class_probs_list, weights, args.iou_thr)
            if voted is not None:
                for i, v in enumerate(voted[: args.max_det]):
                    if v >= 0:
                        labels[i] = float(v)

        # [0, 1] → 원본 픽셀 복원
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= orig_w
            boxes[:, [1, 3]] *= orig_h

        final_predictions.append({
            "image_id": image_id,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "scores": torch.tensor(scores, dtype=torch.float32),
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    make_submission(
        predictions=final_predictions,
        label_to_category=label_to_category,
        output_path=output_path,
    )
    print(f"저장 완료: {output_path}")


main()
