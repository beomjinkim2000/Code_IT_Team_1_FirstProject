import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PillDataset
from src.data.transforms import val_transform
from src.engine.postprocess import PostprocessConfig, postprocess_raw_outputs
from src.engine.predict import predict_batch
from src.models.baseline import build_model
from src.submission.make_submission import make_submission
from src.utils.collate import collate_fn
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="경구약제 객체 탐지 예측 및 제출 파일 생성")
    parser.add_argument("--config", default="configs/default.yaml", help="config 파일 경로")
    parser.add_argument("--checkpoint", default=None, help="체크포인트 경로 (미지정 시 best_model.pt)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    img_size = cfg["train"]["img_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    checkpoint_dir = Path(cfg["paths"]["checkpoint"])
    if args.checkpoint:
        ckpt_path = args.checkpoint
    elif cfg["ema"]["enabled"] and (checkpoint_dir / "best_model_ema.pt").exists():
        ckpt_path = checkpoint_dir / "best_model_ema.pt"
    else:
        ckpt_path = checkpoint_dir / "best_model.pt"
    print(f"체크포인트 로드: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    model = build_model(cfg["data"]["nc"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    test_ds = PillDataset(split="test", transforms=val_transform(img_size))
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    postprocess_cfg = PostprocessConfig(**cfg["postprocess"])
    all_predictions = []

    for images, targets in tqdm(test_loader, desc="predict"):
        batch = torch.stack(images).to(device)
        image_ids = [t["image_id"] for t in targets]
        raw = predict_batch(model, batch, device)
        preds = postprocess_raw_outputs(raw, image_ids=image_ids, config=postprocess_cfg)

        # 모델 입력(img_size×img_size) 좌표 → 원본 이미지 좌표로 스케일백
        for pred, target in zip(preds, targets):
            orig_h, orig_w = target["original_size"]
            scale_x = orig_w / img_size
            scale_y = orig_h / img_size
            if len(pred["boxes"]) > 0:
                pred["boxes"][:, [0, 2]] *= scale_x
                pred["boxes"][:, [1, 3]] *= scale_y

        all_predictions.extend(preds)

    # predictions.json 저장
    pred_dir = Path(cfg["paths"]["prediction"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_json_path = pred_dir / "predictions.json"
    pred_json_path.write_text(
        json.dumps(
            [
                {
                    "image_id": p["image_id"],
                    "boxes": p["boxes"].tolist(),
                    "labels": p["labels"].tolist(),
                    "scores": p["scores"].tolist(),
                }
                for p in all_predictions
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"predictions.json 저장: {pred_json_path}")

    # submission.csv 저장
    submission_path = Path(cfg["paths"]["submission"]) / "submission_v1.csv"
    make_submission(
        predictions=all_predictions,
        label_to_category=cfg["data"]["label_to_category"],
        output_path=submission_path,
    )
    print(f"submission.csv 저장: {submission_path}")



