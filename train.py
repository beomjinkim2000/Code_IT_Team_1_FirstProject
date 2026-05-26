import argparse
import csv
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics.utils.loss import v8DetectionLoss

from src.data.dataset import PillDataset, RAW_DATA_ROOT
from src.data.split import train_val_split
from src.data.transforms import train_transform, val_transform
from src.engine.checkpoint import save_checkpoint
from src.engine.evaluate import evaluate
from src.engine.postprocess import PostprocessConfig, postprocess_raw_outputs
from src.engine.predict import predict_batch
from src.engine.train import _prepare_loss_args, train_one_epoch
from src.models.baseline import build_model, freeze_except_cv3_last, unfreeze_head, unfreeze_all
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


def _make_phase1_optimizer(model, phase1_lr):
    """Phase 1: cv3 마지막 Conv2d만 학습."""
    params = [p for branch in model.model[-1].cv3 for p in branch[-1].parameters()]
    return torch.optim.AdamW(params, lr=phase1_lr)


def _make_phase2_optimizer(model, phase2_lr):
    """Phase 2: backbone/neck frozen 유지, Detect head 전체 학습."""
    params = list(model.model[-1].parameters())
    return torch.optim.AdamW(params, lr=phase2_lr)


def _make_phase3_optimizer(model, head_lr, backbone_lr):
    """Phase 3: backbone/neck, head 각각 별도 lr로 전체 fine-tune."""
    backbone_neck_params = [p for layer in list(model.model)[:-1] for p in layer.parameters()]
    head_params = list(model.model[-1].parameters())
    return torch.optim.AdamW([
        {"params": backbone_neck_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ])


def main():
    parser = argparse.ArgumentParser(description="경구약제 객체 탐지 학습")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="config 파일 경로"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    img_size = cfg["train"]["img_size"]
    phase1_lr = cfg["train"].get("phase1_lr", 0.001)
    phase1_lr_min = cfg["train"].get("phase1_lr_min", 0.00001)
    phase2_lr = cfg["train"].get("phase2_lr", 0.001)
    phase2_lr_min = cfg["train"].get("phase2_lr_min", 0.00001)
    phase3_head_lr = cfg["train"].get("phase3_head_lr", 0.001)
    phase3_backbone_lr = cfg["train"].get("phase3_backbone_lr", 0.00001)
    phase3_lr_min = cfg["train"].get("phase3_lr_min", 0.00001)
    total_epochs = cfg["train"]["epochs"]
    freeze_epochs = max(1, int(total_epochs * cfg["train"].get("freeze_ratio", 0.2)))
    finetune_epochs = max(1, int(total_epochs * cfg["train"].get("finetune_ratio", 0.4)))
    unfreeze_mode = cfg["train"].get("unfreeze_mode", "head")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = build_model(cfg["data"]["nc"])
    freeze_except_cv3_last(model)
    model.to(device)

    annotations = PillDataset.load_annotations()
    category_to_label = cfg["data"]["category_to_label"]

    all_image_files = sorted((RAW_DATA_ROOT / "train_images").glob("*.png"))
    train_files, val_files = train_val_split(
        all_image_files,
        val_ratio=cfg["train"]["val_ratio"],
        seed=cfg["train"]["seed"],
    )
    print(f"train: {len(train_files)}장 / val: {len(val_files)}장")

    train_ds = PillDataset(
        split="train",
        annotations=annotations,
        category_to_label=category_to_label,
        transforms=train_transform(img_size, cfg.get("augmentation")),
        image_files=train_files,
    )
    val_ds = PillDataset(
        split="val",
        annotations=annotations,
        category_to_label=category_to_label,
        transforms=val_transform(img_size),
        image_files=val_files,
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

    # Phase 1 시작: cv3 마지막만 학습
    optimizer = _make_phase1_optimizer(model, phase1_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=freeze_epochs, eta_min=phase1_lr_min)

    _prepare_loss_args(model)
    criterion = v8DetectionLoss(model)
    postprocess_cfg = PostprocessConfig(**cfg["postprocess"])
    eval_postprocess_cfg = PostprocessConfig(
        conf_threshold=cfg["eval"]["conf_threshold"],
        iou_threshold=cfg["postprocess"]["iou_threshold"],
        max_detections=cfg["postprocess"]["max_detections"],
    )

    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "metrics.csv"
    log_file = log_path.open("w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=["epoch", "train_loss", "box_loss", "cls_loss", "dfl_loss", "val_mAP", "val_mAP_50", "lr"])
    log_writer.writeheader()

    best_mAP = -1.0
    for epoch in range(1, total_epochs + 1):

        if epoch == freeze_epochs + 1:
            if unfreeze_mode == "head":
                unfreeze_head(model)
                optimizer = _make_phase2_optimizer(model, phase2_lr)
                print(f"[{epoch:03d}] Phase 2 시작: Detect head 전체 학습 (unfreeze_mode=head)")
            else:
                optimizer = _make_phase1_optimizer(model, phase2_lr)
                print(f"[{epoch:03d}] Phase 2 시작: cv3 마지막 유지 (unfreeze_mode=cv3_last)")
            scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=phase2_lr_min)

        elif epoch == freeze_epochs + finetune_epochs + 1:
            # Phase 3: backbone/neck까지 전체 fine-tune
            unfreeze_all(model)
            optimizer = _make_phase3_optimizer(model, phase3_head_lr, phase3_backbone_lr)
            remaining = total_epochs - freeze_epochs - finetune_epochs
            scheduler = CosineAnnealingLR(optimizer, T_max=max(remaining, 1), eta_min=phase3_lr_min)
            print(f"[{epoch:03d}] Phase 3 시작: backbone/neck lr={phase3_backbone_lr:.6f}, head lr={phase3_head_lr:.6f}")

        model.train()
        train_loss, box_loss, cls_loss, dfl_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        model.eval()
        predictions, targets = _collect_val_predictions(
            model, val_loader, device, eval_postprocess_cfg
        )
        eval_result = evaluate(predictions, targets)
        val_mAP = eval_result["mAP"]
        val_mAP_50 = eval_result["mAP_50"]

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

        current_lr = scheduler.get_last_lr()[0]
        log_writer.writerow({
            "epoch": epoch, "train_loss": round(train_loss, 6),
            "box_loss": round(box_loss, 6), "cls_loss": round(cls_loss, 6), "dfl_loss": round(dfl_loss, 6),
            "val_mAP": round(val_mAP, 6), "val_mAP_50": round(val_mAP_50, 6), "lr": round(current_lr, 8),
        })
        log_file.flush()
        print(
            f"[{epoch:03d}/{total_epochs:03d}] loss: {train_loss:.4f}  box: {box_loss:.4f}  cls: {cls_loss:.4f}  dfl: {dfl_loss:.4f}"
            f"  mAP: {val_mAP:.4f}  mAP@50: {val_mAP_50:.4f}  lr: {current_lr:.6f}"
        )

    log_file.close()
    print(f"학습 완료. best_mAP: {best_mAP:.4f}")


if __name__ == "__main__":
    main()
