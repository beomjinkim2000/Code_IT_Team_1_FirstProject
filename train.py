import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics.utils.loss import v8DetectionLoss

from src.data.dataset import PillDataset
from src.data.transforms import train_transform, val_transform
from src.engine.checkpoint import save_checkpoint
from src.engine.evaluate import evaluate
from src.engine.postprocess import PostprocessConfig, postprocess_raw_outputs
from src.engine.predict import predict_batch
from src.engine.train import _prepare_loss_args, train_one_epoch
from src.models.baseline import build_model
from src.utils.collate import collate_fn
from src.utils.config import load_config
from src.utils.seed import set_seed


def _collect_val_predictions(model, val_loader, device, postprocess_cfg):
    all_predictions = []
    all_targets = []
    for images, targets in tqdm(val_loader, desc="evaluate", leave=False):
        batch = torch.stack(images).to(device)
        raw = predict_batch(model, batch, device)
        image_ids = [t["image_id"] for t in targets]
        preds = postprocess_raw_outputs(
            raw, image_ids=image_ids, config=postprocess_cfg
        )
        all_predictions.extend(preds)
        all_targets.extend(
            {"boxes": t["boxes"], "labels": t["labels"]} for t in targets
        )
    return all_predictions, all_targets


def main():
    parser = argparse.ArgumentParser(description="경구약제 객체 탐지 학습")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="config 파일 경로"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    img_size = cfg["train"]["img_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = build_model(cfg["data"]["nc"])
    model.to(device)

    annotations = PillDataset.load_annotations()
    category_to_label = cfg["data"]["category_to_label"]

    train_ds = PillDataset(
        split="train",
        annotations=annotations,
        category_to_label=category_to_label,
        transforms=train_transform(img_size),
    )
    val_ds = PillDataset(
        split="val",
        annotations=annotations,
        category_to_label=category_to_label,
        transforms=val_transform(img_size),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])

    _prepare_loss_args(model)
    criterion = v8DetectionLoss(model)
    postprocess_cfg = PostprocessConfig(**cfg["postprocess"])
    eval_postprocess_cfg = PostprocessConfig(
        conf_threshold=cfg["eval"]["conf_threshold"],
        iou_threshold=cfg["postprocess"]["iou_threshold"],
        max_detections=cfg["postprocess"]["max_detections"],
    )

    best_mAP = -1.0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        train_loss, box_loss, cls_loss, dfl_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        model.eval()
        predictions, targets = _collect_val_predictions(
            model, val_loader, device, eval_postprocess_cfg
        )
        val_mAP = evaluate(predictions, targets)["mAP"]

        is_best = val_mAP > best_mAP
        if is_best:
            best_mAP = val_mAP

        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_mAP,
            checkpoint_dir=cfg["paths"]["checkpoint"],
            is_best=is_best,
        )

        print(
            f"[{epoch:03d}/{cfg['train']['epochs']:03d}] loss: {train_loss:.4f}  box: {box_loss:.4f}  cls: {cls_loss:.4f}  dfl: {dfl_loss:.4f}  val_mAP: {val_mAP:.4f}"
        )

    print(f"학습 완료. best_mAP: {best_mAP:.4f}")


if __name__ == "__main__":
    main()
