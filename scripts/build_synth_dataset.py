#!/usr/bin/env python3
"""
세그멘테이션 기반 합성 알약 데이터셋 생성 스크립트.

Pipeline
--------
  Phase 1 — 크롭 : GrabCut으로 개별 알약 분리 → pill_bank 적재
  Phase 2 — 배경  : 학습 이미지에서 알약 없는 패치 수집 → bg_pool 적재
  Phase 3 — 합성  : 배경 + 1~4개 알약 조합 → data/augmented/synth/

Usage (Colab)
-------------
  python scripts/build_synth_dataset.py \\
      --data_root data/raw/ai11-level1-project/sprint_ai_project1_data \\
      --out_dir   data/augmented \\
      --n_synth   3000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import wandb
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import PillDataset


# ── Phase 1: GrabCut 알약 세그멘테이션 ───────────────────────────────────────

def grabcut_crop(
    image: np.ndarray,
    bbox_xywh: list,
    padding: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    GrabCut으로 bbox 영역 알약 분리.
    반환: (crop_bgr, mask_uint8)  — mask는 255=foreground, 0=background
    """
    x, y, w, h = [int(v) for v in bbox_xywh]
    H, W = image.shape[:2]

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(W, x + w + padding)
    y2 = min(H, y + h + padding)

    crop = image[y1:y2, x1:x2].copy()
    ch, cw = crop.shape[:2]

    # GrabCut은 최소 크기 필요 — 작으면 bbox 영역 그대로 마스킹
    if ch < 8 or cw < 8:
        mask = np.zeros((ch, cw), np.uint8)
        rx, ry = x - x1, y - y1
        mask[ry : ry + h, rx : rx + w] = 255
        return crop, mask

    gc_mask = np.zeros((ch, cw), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    rx = max(1, x - x1)
    ry = max(1, y - y1)
    rw = min(w, cw - rx - 1)
    rh = min(h, ch - ry - 1)

    try:
        cv2.grabCut(crop, gc_mask, (rx, ry, rw, rh), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((gc_mask == 2) | (gc_mask == 0), 0, 255).astype(np.uint8)
        if fg_mask.sum() < 200:          # GrabCut이 전부 배경 처리 → bbox fallback
            raise ValueError("empty mask")
    except Exception:
        fg_mask = np.zeros((ch, cw), np.uint8)
        fg_mask[ry : ry + rh, rx : rx + rw] = 255

    return crop, fg_mask


# ── Phase 2: 배경 패치 수집 ──────────────────────────────────────────────────

def collect_bg_patches(
    image: np.ndarray,
    bboxes_xywh: list,
    patch_size: int = 320,
    n: int = 3,
) -> list[np.ndarray]:
    """알약 bbox 바깥 영역에서 알약 없는 패치를 최대 n개 수집."""
    H, W = image.shape[:2]
    margin = 24
    pill_mask = np.zeros((H, W), np.uint8)
    for bx, by, bw, bh in bboxes_xywh:
        px1 = max(0, int(bx) - margin)
        py1 = max(0, int(by) - margin)
        px2 = min(W, int(bx + bw) + margin)
        py2 = min(H, int(by + bh) + margin)
        pill_mask[py1:py2, px1:px2] = 1

    patches: list[np.ndarray] = []
    if W <= patch_size or H <= patch_size:
        return patches
    for _ in range(n * 12):
        if len(patches) >= n:
            break
        rx = random.randint(0, W - patch_size)
        ry = random.randint(0, H - patch_size)
        if pill_mask[ry : ry + patch_size, rx : rx + patch_size].sum() == 0:
            patches.append(image[ry : ry + patch_size, rx : rx + patch_size].copy())
    return patches


# ── Phase 3: 합성 이미지 생성 ────────────────────────────────────────────────

def _resize_rotate(
    crop: np.ndarray,
    mask: np.ndarray,
    scale: float,
    angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    ph, pw = crop.shape[:2]
    pw = max(4, int(pw * scale))
    ph = max(4, int(ph * scale))
    crop = cv2.resize(crop, (pw, ph), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_LINEAR)

    if abs(angle) > 0.5:
        M = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
        crop = cv2.warpAffine(crop, M, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.warpAffine(mask, M, (pw, ph), flags=cv2.INTER_NEAREST)
    return crop, mask


def paste_pill(
    canvas: np.ndarray,
    crop: np.ndarray,
    mask: np.ndarray,
    cx: int,
    cy: int,
    scale: float = 1.0,
    angle: float = 0.0,
) -> tuple[np.ndarray, list | None]:
    """
    canvas 위에 crop을 (cx, cy) 중심으로 alpha-blend 합성.
    반환: (canvas, bbox_xywh) — 배치 실패 시 bbox=None
    """
    crop, mask = _resize_rotate(crop, mask, scale, angle)
    ph, pw = crop.shape[:2]
    H, W = canvas.shape[:2]

    x1, y1 = cx - pw // 2, cy - ph // 2
    x2, y2 = x1 + pw, y1 + ph
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(W, x2), min(H, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return canvas, None

    px1, py1_ = cx1 - x1, cy1 - y1
    px2, py2_ = px1 + (cx2 - cx1), py1_ + (cy2 - cy1)

    alpha = (mask[py1_ : py2_, px1:px2] > 127).astype(np.float32)[..., np.newaxis]
    region = canvas[cy1:cy2, cx1:cx2]
    canvas[cy1:cy2, cx1:cx2] = (
        crop[py1_ : py2_, px1:px2] * alpha + region * (1 - alpha)
    ).astype(np.uint8)

    return canvas, [cx1, cy1, cx2 - cx1, cy2 - cy1]


def _boxes_overlap(a: tuple, b: tuple, iou_thr: float = 0.0) -> bool:
    """a, b = (x1, y1, x2, y2). IOU 기반 겹침 판단 (iou_thr=0 → 접촉도 겹침 처리)."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    if iou_thr == 0.0:
        return True
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1) > iou_thr


def make_synth_image(
    pill_bank: list[dict],
    pill_weights: list[float],
    bg_pool: list[np.ndarray],
    img_size: int = 640,
    pill_counts: list[int] | None = None,
) -> tuple[np.ndarray, list[dict]] | None:
    """
    배경 + pill_counts 중 하나를 선택한 수만큼 알약 조합으로 합성 이미지 1장 생성.
    pill_weights: 희귀 클래스 과샘플링용 역빈도 가중치 (pill_bank 인덱스별)
    반환: (image_bgr, [{bbox_xywh, category_id}, ...])  | None (배치 불가)
    """
    if pill_counts is None:
        pill_counts = [3, 4]
    n_pills = random.choice(pill_counts)
    pills = random.choices(pill_bank, weights=pill_weights, k=n_pills)

    if bg_pool:
        bg = cv2.resize(
            random.choice(bg_pool), (img_size, img_size), interpolation=cv2.INTER_LINEAR
        )
    else:
        # 배경 없으면 밝은 회색 단색 (실험실 바닥 근사)
        val = random.randint(180, 230)
        bg = np.full((img_size, img_size, 3), val, dtype=np.uint8)

    canvas = bg.copy()
    placed_xyxy: list[tuple] = []
    annotations: list[dict] = []

    for pill_info in pills:
        scale = random.uniform(0.5, 1.3)
        angle = random.uniform(-180, 180)

        ph, pw = pill_info["crop"].shape[:2]
        pw_s = max(4, int(pw * scale))
        ph_s = max(4, int(ph * scale))
        margin = 10

        placed = False
        for _ in range(15):
            cx = random.randint(pw_s // 2 + margin, img_size - pw_s // 2 - margin)
            cy = random.randint(ph_s // 2 + margin, img_size - ph_s // 2 - margin)
            new_box = (cx - pw_s // 2, cy - ph_s // 2, cx + pw_s // 2, cy + ph_s // 2)

            if any(_boxes_overlap(new_box, ob) for ob in placed_xyxy):
                continue

            canvas, bbox = paste_pill(
                canvas, pill_info["crop"], pill_info["mask"], cx, cy, scale, angle
            )
            if bbox is not None:
                annotations.append({"bbox_xywh": bbox, "category_id": pill_info["category_id"]})
                placed_xyxy.append(new_box)
                placed = True
                break

        # 배치 실패한 알약은 스킵

    return (canvas, annotations) if annotations else None


# ── 메인 ─────────────────────────────────────────────────────────────────────

def build(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    synth_img_dir = out_dir / "synth" / "images"
    synth_img_dir.mkdir(parents=True, exist_ok=True)

    # ── WandB 초기화 (WANDB_API_KEY 없으면 disabled) ──
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "health-eat-pill-detection"),
        project=os.environ.get("WANDB_PROJECT", "health-eat-pill-detection"),
        name="synth-build",
        job_type="data-augmentation",
        config=vars(args),
        mode="online" if use_wandb else "disabled",
    )

    # ── 어노테이션 로드 ──
    print("어노테이션 로드 중...")
    annotations = PillDataset.load_annotations(data_root)
    image_dir = data_root / "train_images"

    pill_bank: list[dict] = []
    bg_pool: list[np.ndarray] = []

    print(f"{len(annotations)}장 이미지 처리 중 (GrabCut + 배경 수집)...")
    for fname, ann in tqdm(annotations.items(), desc="Phase 1+2"):
        img_path = image_dir / fname
        if not img_path.exists():
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        bboxes = ann["boxes_xywh"]
        labels = ann["labels"]

        for bbox, cat_id in zip(bboxes, labels):
            crop, mask = grabcut_crop(image, bbox, padding=args.padding)
            pill_bank.append({"crop": crop, "mask": mask, "category_id": cat_id})

        bg_pool.extend(
            collect_bg_patches(image, bboxes, patch_size=args.bg_patch_size, n=args.n_bg_per_image)
        )

    print(f"  알약 크롭 {len(pill_bank)}개 | 배경 패치 {len(bg_pool)}개")
    wandb.log({"phase1_crops": len(pill_bank), "phase2_backgrounds": len(bg_pool)})

    if not pill_bank:
        print("[ERROR] 크롭 없음. --data_root 경로를 확인하세요.")
        wandb.finish(exit_code=1)
        return

    # pill_bank 샘플 20개 저장 (노트북 시각화용)
    sample_dir = out_dir / "synth" / "pill_bank_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, len(pill_bank) // 20)
    for i, pill in enumerate(pill_bank[::step][:20]):
        fname = f"{i:02d}_cat{pill['category_id']}.png"
        cv2.imwrite(str(sample_dir / fname), pill["crop"])

    # 희귀 클래스 과샘플링: 클래스 등장 빈도의 역수를 가중치로 사용
    cat_freq = Counter(p["category_id"] for p in pill_bank)
    max_freq = max(cat_freq.values())
    pill_weights = [max_freq / cat_freq[p["category_id"]] for p in pill_bank]
    print(f"  클래스 {len(cat_freq)}종 | 최소 {min(cat_freq.values())}개 ~ 최대 {max_freq}개 (역빈도 가중치 적용)")

    # ── Phase 3: 합성 ──
    print(f"합성 이미지 {args.n_synth}장 생성 중... (알약 수: {args.pill_counts})")
    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    ann_id = 1
    id_offset = 1_000_000   # 실제 image_id(0~수만)와 충돌 방지
    cat_counter: Counter = Counter()
    saved = 0  # 성공 저장 수 — i(시도 횟수)와 분리해 id/파일명 gap 방지

    for _ in tqdm(range(args.n_synth), desc="Phase 3"):
        result = make_synth_image(
            pill_bank, pill_weights, bg_pool,
            img_size=args.img_size,
            pill_counts=args.pill_counts,
        )
        if result is None:
            continue

        canvas, pill_anns = result
        image_id = id_offset + saved
        fname = f"synth_{saved:06d}.png"
        saved += 1
        cv2.imwrite(str(synth_img_dir / fname), canvas)

        coco_images.append({
            "id": image_id,
            "file_name": fname,
            "width": args.img_size,
            "height": args.img_size,
        })
        for pa in pill_anns:
            x, y, w, h = pa["bbox_xywh"]
            coco_annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": pa["category_id"],
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            cat_counter[pa["category_id"]] += 1
            ann_id += 1

    # COCO JSON 저장 (SynthPillDataset 호환)
    ann_path = out_dir / "synth" / "annotations.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(
            {"images": coco_images, "annotations": coco_annotations, "categories": []},
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ── WandB 최종 요약 + artifact ──
    n_images = len(coco_images)
    n_anns = len(coco_annotations)
    wandb.log({
        "synth_images": n_images,
        "synth_annotations": n_anns,
        "avg_pills_per_image": n_anns / max(n_images, 1),
        "unique_categories": len(cat_counter),
    })

    # class별 분포 테이블
    class_table = wandb.Table(columns=["category_id", "count"])
    for cat_id, cnt in sorted(cat_counter.items()):
        class_table.add_data(cat_id, cnt)
    wandb.log({"synth_class_distribution": class_table})

    # synth 디렉토리 전체(이미지 + annotations.json)를 WandB artifact로 업로드
    # → 다음 Colab 세션에서 SynthPillDataset이 자동으로 다운로드해 재사용
    if use_wandb:
        artifact = wandb.Artifact(
            name="synth-dataset",
            type="dataset",
            metadata={"n_images": n_images, "n_annotations": n_anns, "seed": args.seed},
        )
        artifact.add_dir(str(out_dir / "synth"), name="synth")
        wandb.log_artifact(artifact)
        print("WandB artifact 'synth-dataset' 업로드 완료")

    wandb.finish()
    print(f"\n완료! 이미지 {n_images}장 | 어노테이션 {n_anns}개")
    print(f"저장 경로: {out_dir / 'synth'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="세그멘테이션 기반 합성 알약 데이터셋 생성")
    parser.add_argument(
        "--data_root",
        default="data/raw/ai11-level1-project/sprint_ai_project1_data",
        help="원본 데이터 루트 (train_images/, train_annotations/ 포함)",
    )
    parser.add_argument("--out_dir", default="data/augmented", help="합성 데이터 출력 폴더")
    parser.add_argument("--n_synth", type=int, default=3000, help="생성할 합성 이미지 수")
    parser.add_argument("--img_size", type=int, default=640, help="합성 이미지 해상도")
    parser.add_argument(
        "--pill_counts", type=int, nargs="+", default=[3, 4],
        help="이미지당 알약 수 후보 (랜덤 선택). 예: --pill_counts 3 4",
    )
    parser.add_argument("--padding", type=int, default=12, help="GrabCut 크롭 여백 픽셀")
    parser.add_argument("--bg_patch_size", type=int, default=320, help="배경 패치 크기")
    parser.add_argument("--n_bg_per_image", type=int, default=3, help="이미지당 배경 패치 수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    build(args)


if __name__ == "__main__":
    main()
