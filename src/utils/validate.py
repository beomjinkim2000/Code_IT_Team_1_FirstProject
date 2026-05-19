import random
import warnings
import importlib
from typing import List, Tuple, Dict

import torch
from torch import Tensor
from torch.utils.data import Dataset

REQUIRED_PACKAGES = [
    "torch", "torchvision", "ultralytics",
    "cv2", "albumentations", "PIL", "pandas", "yaml",
]


# ── 패키지 설치 확인 ──────────────────────────────────────────────────────────

def check_packages() -> None:
    missing = [pkg for pkg in REQUIRED_PACKAGES if not _importable(pkg)]
    if missing:
        raise ImportError(f"미설치 패키지: {missing}\n  → uv sync 또는 pip install 실행")
    print(f"[패키지] ✅ 필수 패키지 {len(REQUIRED_PACKAGES)}개 설치 확인")


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


# ── 이미지 단일 검사 ──────────────────────────────────────────────────────────

def _check_image(image: Tensor, idx: int) -> Tuple[List[str], List[str]]:
    """
    Returns (errors, warnings).
    errors   → 필수 조건 위반, 학습 불가
    warnings → 권장 조건 미충족, 학습 성능에 영향 가능
    """
    errors: List[str] = []
    warns: List[str] = []

    # 1. torch.Tensor인가?
    if not isinstance(image, Tensor):
        errors.append(f"[idx={idx}] torch.Tensor가 아님: {type(image)}")
        return errors, warns  # 이후 체크 의미 없음

    # 2. [C,H,W] 또는 [B,C,H,W]인가?
    if image.ndim == 3:
        C, H, W = image.shape
    elif image.ndim == 4:
        _, C, H, W = image.shape
    else:
        errors.append(f"[idx={idx}] 차원이 [C,H,W] 또는 [B,C,H,W]가 아님: ndim={image.ndim}")
        return errors, warns

    # 3. 채널 수 3인가?
    if C != 3:
        errors.append(f"[idx={idx}] 채널 수가 3이 아님: C={C}")

    # 4. H, W가 32배수인가?
    if H % 32 != 0 or W % 32 != 0:
        errors.append(f"[idx={idx}] H, W가 32의 배수가 아님: {H}x{W}")

    # 5. dtype float32? (권장)
    if image.dtype != torch.float32:
        warns.append(f"[idx={idx}] dtype이 float32가 아님: {image.dtype}")

    # 6. 값 범위 0~1? (권장)
    vmin, vmax = image.min().item(), image.max().item()
    if vmin < 0.0 or vmax > 1.0:
        warns.append(f"[idx={idx}] 값 범위가 0~1 벗어남: min={vmin:.3f}, max={vmax:.3f} (정규화 확인)")

    return errors, warns


# ── 2단계 데이터셋 전체 검증 ──────────────────────────────────────────────────

def validate_dataset(dataset: Dataset, img_size: int, sample_size: int = 10) -> None:
    """
    Stage 1: 랜덤 sample_size장 검사 → 실패 시 즉시 중단
    Stage 2: Stage 1 통과 시 전체 검사
    """
    check_packages()

    total = len(dataset)
    sample_indices = random.sample(range(total), min(sample_size, total))

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    print(f"\n[Stage 1] 랜덤 {len(sample_indices)}장 검증 중...")
    _run_checks(dataset, sample_indices, img_size, stage=1)
    print(f"[Stage 1] ✅ 통과 → 전체 {total}장 검증 시작\n")

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    print(f"[Stage 2] 전체 {total}장 검증 중...")
    _run_checks(dataset, range(total), img_size, stage=2)
    print(f"[Stage 2] ✅ 전체 {total}장 통과\n")


def _run_checks(dataset: Dataset, indices, img_size: int, stage: int) -> None:
    all_warns: List[str] = []

    for idx in indices:
        image, _ = dataset[idx]
        errors, warns = _check_image(image, idx)
        all_warns.extend(warns)

        if errors:
            print(f"\n[Stage {stage}] ❌ 검증 실패")
            for msg in errors:
                print(f"  {msg}")
            raise AssertionError(f"Stage {stage} 검증 실패 (idx={idx}) — 전처리 코드 확인 필요")

    for msg in all_warns:
        warnings.warn(msg)


# ── 배치 단위 검증 (engine/train.py 진입 전용) ────────────────────────────────

def validate_batch(images: List[Tensor], targets: List[Dict], img_size: int) -> None:
    """학습 루프 첫 배치에서 빠르게 확인. validate_dataset 통과 후에도 호출 권장."""
    assert len(images) == len(targets), \
        f"images({len(images)})와 targets({len(targets)}) 개수 불일치"

    for i, (image, target) in enumerate(zip(images, targets)):
        errors, warns = _check_image(image, i)
        for msg in warns:
            warnings.warn(msg)
        if errors:
            raise AssertionError("\n".join(errors))

        for key in ("boxes", "labels", "image_id"):
            assert key in target, f"[{i}] target에 '{key}' 키 없음"

        boxes  = target["boxes"]
        labels = target["labels"]

        assert isinstance(boxes, Tensor) and boxes.ndim == 2 and boxes.shape[1] == 4, \
            f"[{i}] boxes shape이 [N,4]가 아님: {boxes.shape}"
        assert boxes.dtype == torch.float32, \
            f"[{i}] boxes dtype이 float32가 아님: {boxes.dtype}"
        if boxes.shape[0] > 0:
            assert (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all(), \
                f"[{i}] boxes가 xyxy 형식이 아님 (x2>x1, y2>y1 불만족)"

        assert isinstance(labels, Tensor) and labels.ndim == 1, \
            f"[{i}] labels shape이 [N]이 아님"
        assert labels.dtype == torch.int64, \
            f"[{i}] labels dtype이 int64가 아님: {labels.dtype}"
        assert labels.shape[0] == boxes.shape[0], \
            f"[{i}] boxes({boxes.shape[0]})와 labels({labels.shape[0]}) 개수 불일치"
