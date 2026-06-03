"""
Google Drive에서 읽은 metrics CSV → WandB epoch-by-epoch 업로드
Downloads/best_model.pt → WandB artifact (v5-11x-250)

사용법:
  python scripts/upload_metrics_to_wandb.py
  python scripts/upload_metrics_to_wandb.py --dry-run   # WandB 연결 없이 확인만
"""
import argparse
import csv
import os
from pathlib import Path

ENTITY = os.environ.get("WANDB_ENTITY", "health-eat-pill-detection")
PROJECT = os.environ.get("WANDB_PROJECT", "health-eat-pill-detection")

CSV_DIR = Path("/tmp/health_eat_metrics")
MODEL_PATH = Path("/Users/apple/Downloads/best_model.pt")

RUNS = [
    {
        "name": "v2.2-yolov8n-freeze",
        "model": "yolov8n",
        "epochs": 50,
        "batch_size": 8,
        "tags": ["v2.2", "yolov8n", "freeze", "historical"],
        "csv_file": "metricsV2.2.csv",
        "fmt": "v1",
        "kaggle_score": 0.534,
        "artifact_path": None,
        "notes": "Baseline freeze training. Phase1=freeze(ep1-10), Phase2=unfreeze(ep11-30), Phase3=warmup-lr(ep31-50).",
    },
    {
        "name": "v2.5-yolov8x-3phase",
        "model": "yolov8x",
        "epochs": 200,
        "batch_size": 8,
        "tags": ["v2.5", "yolov8x", "3phase", "historical"],
        "csv_file": "metricsV2.5.csv",
        "fmt": "v2",
        "kaggle_score": 0.723,
        "artifact_path": None,
        "notes": "YOLOv8x 3-phase lr schedule. Phase1=freeze-backbone(ep1-40), Phase2=head-only(ep41-128), Phase3=full-finetune(ep129-200).",
    },
    {
        "name": "v2.7-yolov8x-aug",
        "model": "yolov8x",
        "epochs": 100,
        "batch_size": 8,
        "tags": ["v2.7", "yolov8x", "augmentation", "historical"],
        "csv_file": "metricsV2.7.csv",
        "fmt": "v2",
        "kaggle_score": 0.74,
        "artifact_path": None,
        "notes": "YOLOv8x with augmentation. 3-phase training.",
    },
    {
        "name": "v4-yolo11x-3phase",
        "model": "yolo11x",
        "epochs": 250,
        "batch_size": 8,
        "tags": ["v4", "yolo11x", "3phase", "historical"],
        "csv_file": "metrics_V4.csv",
        "fmt": "v2",
        "kaggle_score": None,
        "artifact_path": None,
        "notes": "YOLO11x 3-phase. V4-1(freeze ep1-70), V4-2(unfreeze ep71-210), V4-3(full ep211-250).",
    },
    {
        "name": "v5-yolo11x-250ep",
        "model": "yolo11x",
        "epochs": 250,
        "batch_size": 8,
        "tags": ["v5", "yolo11x", "best", "historical"],
        "csv_file": "metrics_V5.csv",
        "fmt": "v3",
        "kaggle_score": None,
        "artifact_path": str(MODEL_PATH) if MODEL_PATH.exists() else None,
        "notes": "YOLO11x best model. val mAP@50 EMA=1.0 at ep250.",
    },
]


def parse_csv(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def build_metrics(row: dict, fmt: str) -> dict:
    ep = int(row["epoch"])
    base = {
        "train/loss": float(row["train_loss"]),
        "train/box_loss": float(row["box_loss"]),
        "train/cls_loss": float(row["cls_loss"]),
        "train/dfl_loss": float(row["dfl_loss"]),
        "lr": float(row["lr"]),
    }
    if fmt == "v1":
        base["val/mAP"] = float(row["val_mAP"])
        base["val/mAP50"] = float(row["val_mAP_50"])
    elif fmt in ("v2", "v3"):
        base["val/mAP_raw"] = float(row["val_mAP_raw"])
        base["val/mAP50_raw"] = float(row["val_mAP_50_raw"])
        base["val/mAP_ema"] = float(row["val_mAP_ema"])
        base["val/mAP50_ema"] = float(row["val_mAP_50_ema"])
        if fmt == "v3":
            base["val/box_loss"] = float(row["val_box_loss"])
            base["val/cls_loss"] = float(row["val_cls_loss"])
            base["val/dfl_loss"] = float(row["val_dfl_loss"])
    return base


def upload_run(cfg: dict, dry_run: bool = False):
    csv_path = CSV_DIR / cfg["csv_file"]
    if not csv_path.exists():
        print(f"  [SKIP] CSV 없음: {csv_path}")
        return

    rows = parse_csv(csv_path)
    print(f"  {cfg['name']}: {len(rows)} epochs, fmt={cfg['fmt']}")

    if dry_run:
        final = rows[-1]
        m = build_metrics(final, cfg["fmt"])
        key = "val/mAP_ema" if "val/mAP_ema" in m else "val/mAP"
        print(f"    final {key}={m.get(key, 'N/A'):.4f}")
        return

    import wandb

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=cfg["name"],
        config={
            "model": cfg["model"],
            "epochs": cfg["epochs"],
            "batch_size": cfg["batch_size"],
            "source": "gdrive-historical",
        },
        tags=cfg["tags"],
        notes=cfg.get("notes", ""),
        reinit=True,
    )
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    best_mAP = 0.0
    for row in rows:
        ep = int(row["epoch"])
        metrics = build_metrics(row, cfg["fmt"])
        metrics["epoch"] = ep
        wandb.log(metrics, step=ep)
        mAP = metrics.get("val/mAP_ema", metrics.get("val/mAP", 0.0))
        best_mAP = max(best_mAP, mAP)

    # summary
    wandb.summary["best_mAP_ema"] = best_mAP
    wandb.summary["total_epochs"] = cfg["epochs"]
    if cfg.get("kaggle_score") is not None:
        wandb.summary["kaggle_score"] = cfg["kaggle_score"]

    # artifact
    if cfg.get("artifact_path"):
        artifact_path = Path(cfg["artifact_path"])
        if artifact_path.exists():
            ver = cfg["name"].split("-")[0]  # "v5" 등
            artifact = wandb.Artifact(
                name=f"best-{ver}",
                type="model",
                metadata={
                    "run": cfg["name"],
                    "best_mAP_ema": best_mAP,
                    "kaggle_score": cfg.get("kaggle_score"),
                },
            )
            artifact.add_file(str(artifact_path), name=f"best-{ver}.pt")
            run.log_artifact(artifact)
            print(f"    artifact 업로드: best({ver}).pt ({artifact_path.stat().st_size/1e6:.0f}MB)")
        else:
            print(f"    [WARN] artifact 파일 없음: {artifact_path}")

    run.finish()
    print(f"    완료 — best mAP EMA: {best_mAP:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="WandB 연결 없이 데이터 확인만")
    parser.add_argument("--only", default=None, help="특정 버전만 (예: v5)")
    args = parser.parse_args()

    if not args.dry_run:
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            raise RuntimeError("WANDB_API_KEY 환경변수 필요")

    print(f"업로드 대상: {PROJECT}")
    for cfg in RUNS:
        if args.only and args.only not in cfg["name"]:
            continue
        print(f"\n[{cfg['name']}]")
        upload_run(cfg, dry_run=args.dry_run)

    print("\n모든 업로드 완료!")


if __name__ == "__main__":
    main()
