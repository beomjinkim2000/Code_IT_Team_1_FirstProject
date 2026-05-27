import argparse
import csv
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from ultralytics.utils.loss import v8DetectionLoss

from src.data.dataset import PillDataset, RAW_DATA_ROOT
from src.data.mosaic import MosaicDataset
from src.data.split import build_split_metadata, train_val_split
from src.data.transforms import train_transform, val_transform
from src.engine.checkpoint import save_checkpoint
from src.engine.ema import ModelEMA
from src.engine.evaluate import evaluate
from src.engine.postprocess import PostprocessConfig, postprocess_raw_outputs
from src.engine.predict import predict_batch
from src.engine.train import _prepare_loss_args, train_one_epoch
from src.utils.class_weights import compute_sample_weights
from src.models.baseline import build_model, freeze_except_cv3_last, set_frozen_bn_eval, unfreeze_head, unfreeze_all
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


def _make_optimizer(opt_cfg, param_groups):
    """opt_cfg['type'] 에 따라 AdamW 또는 SGD를 생성한다."""
    opt_type = opt_cfg.get("type", "adamw").lower()
    if opt_type == "sgd":
        return torch.optim.SGD(
            param_groups,
            momentum=opt_cfg.get("momentum", 0.937),
            weight_decay=opt_cfg.get("weight_decay", 0.0005),
            nesterov=opt_cfg.get("nesterov", True),
        )
    return torch.optim.AdamW(param_groups)


def _make_phase1_optimizer(model, phase1_lr, opt_cfg):
    """Phase 1: cv3 마지막 Conv2d만 학습."""
    params = [p for branch in model.model[-1].cv3 for p in branch[-1].parameters()]
    return _make_optimizer(opt_cfg, [{"params": params, "lr": phase1_lr}])


def _make_phase2_optimizer(model, phase2_lr, opt_cfg):
    """Phase 2: backbone/neck frozen 유지, Detect head 전체 학습."""
    params = list(model.model[-1].parameters())
    return _make_optimizer(opt_cfg, [{"params": params, "lr": phase2_lr}])


def _make_phase3_optimizer(model, head_lr, backbone_lr, opt_cfg):
    """Phase 3: backbone/neck, head 각각 별도 lr로 전체 fine-tune."""
    backbone_neck_params = [p for layer in list(model.model)[:-1] for p in layer.parameters()]
    head_params = list(model.model[-1].parameters())
    return _make_optimizer(opt_cfg, [
        {"params": backbone_neck_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ])


def _make_phase3_scheduler(optimizer, warmup_epochs, remaining, lr_min):
    """Phase 3용 스케줄러. warmup_epochs > 0이면 LinearLR → CosineAnnealingLR 순서로 연결."""
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=max(remaining, 1), eta_min=lr_min)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(remaining - warmup_epochs, 1), eta_min=lr_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


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
    phase3_head_lr = cfg["train"].get("phase3_head_lr", 0.0001)
    phase3_backbone_lr = cfg["train"].get("phase3_backbone_lr", 0.00001)
    phase3_lr_min = cfg["train"].get("phase3_lr_min", 0.000001)
    phase3_warmup_epochs = cfg["train"].get("phase3_warmup_epochs", 0)
    total_epochs = cfg["train"]["epochs"]
    freeze_epochs = max(1, int(total_epochs * cfg["train"].get("freeze_ratio", 0.2)))
    finetune_epochs = max(1, int(total_epochs * cfg["train"].get("finetune_ratio", 0.4)))
    unfreeze_mode = cfg["train"].get("unfreeze_mode", "head")
    opt_cfg = cfg.get("optimizer", {"type": "adamw"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = build_model(cfg["data"]["nc"])
    freeze_except_cv3_last(model)
    model.to(device)
    ema = None  # Phase 2 시작 시 초기화 (Phase 1은 대부분 frozen → COCO 초기값 평균 방지)

    annotations = PillDataset.load_annotations()
    category_to_label = cfg["data"]["category_to_label"]
    labels_by_id, image_id_by_file = build_split_metadata(annotations, category_to_label)
    split_cfg = cfg.get("split", {})

    all_image_files = sorted((RAW_DATA_ROOT / "train_images").glob("*.png"))
    train_files, val_files = train_val_split(
        all_image_files,
        val_ratio=split_cfg.get("val_ratio", 0.2),
        seed=cfg["train"]["seed"],
        method=split_cfg.get("method", "random"),
        labels_by_id=labels_by_id,
        image_id_by_file=image_id_by_file,
        num_classes=cfg["data"]["nc"],
        output_dir=split_cfg.get("output_dir"),
    )
    print(f"train: {len(train_files)}장 / val: {len(val_files)}장")

    train_ds = PillDataset(
        split="train",
        annotations=annotations,
        category_to_label=category_to_label,
        transforms=train_transform(img_size, cfg.get("augmentation")),
        image_files=train_files,
    )
    aug_cfg = cfg.get("augmentation") or {}
    train_ds = MosaicDataset(
        train_ds,
        img_size=img_size,
        p=aug_cfg.get("mosaic_p", 0.5),
        min_bbox_size=aug_cfg.get("mosaic_min_bbox_size", 2),
    )

    val_ds = PillDataset(
        split="val",
        annotations=annotations,
        category_to_label=category_to_label,
        transforms=val_transform(img_size),
        image_files=val_files,
    )

    cw_cfg = cfg.get("class_weights") or {}
    cw_method = cw_cfg.get("method")
    if cw_method:
        sample_weights = compute_sample_weights(
            image_paths=train_ds.dataset.image_paths,
            annotations=annotations,
            category_to_label=category_to_label,
            num_classes=cfg["data"]["nc"],
            method=cw_method,
            manual=cw_cfg.get("manual"),
        )
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], sampler=sampler, collate_fn=collate_fn)
        print(f"class_weights: method={cw_method}, sample_weights min={min(sample_weights):.3f} max={max(sample_weights):.3f}")
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate_fn)

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    opt_type = opt_cfg.get("type", "adamw").lower()
    print(f"optimizer: {opt_type}")

    # Phase 1 시작: cv3 마지막만 학습
    optimizer = _make_phase1_optimizer(model, phase1_lr, opt_cfg)
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
    log_writer = csv.DictWriter(log_file, fieldnames=[
        "epoch", "train_loss", "box_loss", "cls_loss", "dfl_loss",
        "val_mAP_raw", "val_mAP_50_raw", "val_mAP_ema", "val_mAP_50_ema", "lr",
    ])
    log_writer.writeheader()

    best_mAP = -1.0
    for epoch in range(1, total_epochs + 1):

        if epoch == freeze_epochs + 1:
            if unfreeze_mode == "head":
                unfreeze_head(model)
                optimizer = _make_phase2_optimizer(model, phase2_lr, opt_cfg)
                print(f"[{epoch:03d}] Phase 2 시작: Detect head 전체 학습 (unfreeze_mode=head)")
            else:
                optimizer = _make_phase1_optimizer(model, phase2_lr, opt_cfg)
                print(f"[{epoch:03d}] Phase 2 시작: cv3 마지막 유지 (unfreeze_mode=cv3_last)")
            scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=phase2_lr_min)
            if cfg["ema"]["enabled"]:
                ema = ModelEMA(model).to(device)

        elif epoch == freeze_epochs + finetune_epochs + 1:
            unfreeze_all(model)
            optimizer = _make_phase3_optimizer(model, phase3_head_lr, phase3_backbone_lr, opt_cfg)
            remaining = total_epochs - freeze_epochs - finetune_epochs
            scheduler = _make_phase3_scheduler(optimizer, phase3_warmup_epochs, remaining, phase3_lr_min)
            # Phase 3 진입 시 EMA를 현재 모델 상태로 리셋 — Phase 2 가중치가 끌어당기는 현상 방지
            if cfg["ema"]["enabled"]:
                ema = ModelEMA(model).to(device)
            warmup_info = f", warmup={phase3_warmup_epochs}ep" if phase3_warmup_epochs > 0 else ""
            print(f"[{epoch:03d}] Phase 3 시작: backbone lr={phase3_backbone_lr:.6f}, head lr={phase3_head_lr:.6f}{warmup_info}")

        model.train()
        set_frozen_bn_eval(model)  # frozen BN이 batch 통계 쓰는 버그 방지
        train_loss, box_loss, cls_loss, dfl_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, ema=ema)
        scheduler.step()

        # 원본 모델 검증
        model.eval()
        raw_preds, raw_targets = _collect_val_predictions(model, val_loader, device, eval_postprocess_cfg)
        raw_result = evaluate(raw_preds, raw_targets)
        val_mAP_raw = raw_result["mAP"]
        val_mAP_50_raw = raw_result["mAP_50"]

        # EMA 모델 검증 (EMA 비활성화 시 원본 결과 그대로 사용)
        if ema is not None:
            ema.model.eval()
            ema_preds, ema_targets = _collect_val_predictions(ema.model, val_loader, device, eval_postprocess_cfg)
            ema_result = evaluate(ema_preds, ema_targets)
            val_mAP_ema = ema_result["mAP"]
            val_mAP_50_ema = ema_result["mAP_50"]
        else:
            val_mAP_ema = val_mAP_raw
            val_mAP_50_ema = val_mAP_50_raw

        is_best = val_mAP_ema > best_mAP
        if is_best:
            best_mAP = val_mAP_ema

        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_mAP_ema,
            checkpoint_dir=cfg["paths"]["checkpoint"],
            is_best=is_best,
            ema=ema,
        )

        current_lr = scheduler.get_last_lr()[-1]
        log_writer.writerow({
            "epoch": epoch, "train_loss": round(train_loss, 6),
            "box_loss": round(box_loss, 6), "cls_loss": round(cls_loss, 6), "dfl_loss": round(dfl_loss, 6),
            "val_mAP_raw": round(val_mAP_raw, 6), "val_mAP_50_raw": round(val_mAP_50_raw, 6),
            "val_mAP_ema": round(val_mAP_ema, 6), "val_mAP_50_ema": round(val_mAP_50_ema, 6),
            "lr": round(current_lr, 8),
        })
        log_file.flush()
        print(
            f"[{epoch:03d}/{total_epochs:03d}] loss: {train_loss:.4f}  box: {box_loss:.4f}  cls: {cls_loss:.4f}  dfl: {dfl_loss:.4f}"
            f"  mAP(raw): {val_mAP_raw:.4f}  mAP(ema): {val_mAP_ema:.4f}  lr: {current_lr:.6f}"
        )

    log_file.close()
    print(f"학습 완료. best_mAP(ema): {best_mAP:.4f}")


main()

