from __future__ import annotations

import random
from pathlib import Path


def train_val_split(
    image_files: list[Path],
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """전체 이미지 파일 목록을 train/val로 나눠 파일명 리스트로 반환한다."""
    file_names = [f.name for f in image_files]
    rng = random.Random(seed)
    shuffled = file_names[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    return shuffled[val_count:], shuffled[:val_count]
