from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np


def build_split_metadata(
    annotations: dict[str, dict[str, Any]],
    category_to_label: dict[int, int] | None = None,
) -> tuple[dict[int, set[int]], dict[str, int]]:
    """annotation dict에서 iterative split에 필요한 image_id 기준 label 정보를 만든다."""
    labels_by_id: dict[int, set[int]] = {}
    image_id_by_file: dict[str, int] = {}

    # Dataset annotation은 파일명 기준 dict이므로, split에서는 image_id 기준 라벨 집합으로 다시 정리한다.
    for file_name, item in annotations.items():
        image_id = int(item["image_id"])
        image_id_by_file[file_name] = image_id

        # 학습용 split은 raw category_id가 아니라 모델이 쓰는 class_id 분포를 맞추는 쪽이 목적이다.
        labels = set()
        for label in item.get("labels", []):
            raw_label = int(label)
            if category_to_label is not None and raw_label not in category_to_label:
                raise KeyError(f"category_to_label에 없는 category_id입니다: {raw_label}")
            labels.add(category_to_label[raw_label] if category_to_label is not None else raw_label)
        labels_by_id[image_id] = labels

    return labels_by_id, image_id_by_file


def train_val_split(
    image_files: list[Path],
    val_ratio: float,
    seed: int,
    method: str = "random",
    labels_by_id: dict[int, set[int]] | None = None,
    image_id_by_file: dict[str, int] | None = None,
    num_classes: int | None = None,
    output_dir: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    """
    전체 이미지 파일 목록을 train/val 파일명 리스트로 나눈다.

    output_dir가 주어지면 split 결과 파일과 클래스 분포 비교 로그를 함께 저장한다.
    """
    # Dataset(image_files=...)에 바로 넘길 수 있도록 Path 목록을 파일명 목록으로 변환한다.
    file_names = [path.name for path in image_files]
    _validate_split_inputs(file_names, val_ratio, method)

    # random은 기존 방식 그대로 seed shuffle 후 val_ratio만큼 자른다.
    if method == "random":
        train_files, val_files = _random_split(file_names, val_ratio, seed)
        random_files = None

    # iterative는 멀티라벨 클래스 분포를 맞추기 위해 image_id별 class_id 집합이 필요하다.
    elif method == "iterative":
        if labels_by_id is None or image_id_by_file is None or num_classes is None:
            raise ValueError("iterative split에는 labels_by_id, image_id_by_file, num_classes가 필요합니다.")
        train_files, val_files = _iterative_split(
            file_names=file_names,
            val_ratio=val_ratio,
            seed=seed,
            labels_by_id=labels_by_id,
            image_id_by_file=image_id_by_file,
            num_classes=num_classes,
        )
        random_files = _random_split(file_names, val_ratio, seed)
    else:
        raise ValueError("method는 random 또는 iterative 중 하나여야 합니다.")

    # output_dir가 지정되면 실제 split 파일과 random 대비 분포 비교 로그를 저장한다.
    if output_dir is not None:
        _save_split_outputs(
            output_dir=Path(output_dir),
            method=method,
            val_ratio=val_ratio,
            seed=seed,
            train_files=train_files,
            val_files=val_files,
            random_files=random_files,
            labels_by_id=labels_by_id,
            image_id_by_file=image_id_by_file,
            num_classes=num_classes,
        )

    return train_files, val_files


def _validate_split_inputs(file_names: list[str], val_ratio: float, method: str) -> None:
    # 잘못된 split 입력은 train.py까지 흘러가기 전에 여기서 빠르게 중단한다.
    if not file_names:
        raise ValueError("image_files가 비어 있습니다.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio는 0과 1 사이여야 합니다.")
    if method not in {"random", "iterative"}:
        raise ValueError("method는 random 또는 iterative 중 하나여야 합니다.")


def _random_split(file_names: list[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    """seed를 고정한 기존 random 80:20 방식 split."""
    # 원본 리스트를 건드리지 않도록 복사본을 섞는다.
    rng = random.Random(seed)
    shuffled = file_names[:]
    rng.shuffle(shuffled)

    # 아주 작은 데이터셋에서도 val이 비지 않도록 최소 1개를 보장한다.
    val_count = max(1, int(len(shuffled) * val_ratio))
    val_files = shuffled[:val_count]
    train_files = shuffled[val_count:]
    return train_files, val_files


def _iterative_split(
    file_names: list[str],
    val_ratio: float,
    seed: int,
    labels_by_id: dict[int, set[int]],
    image_id_by_file: dict[str, int],
    num_classes: int,
) -> tuple[list[str], list[str]]:
    """멀티라벨 클래스 분포를 맞추는 iterative stratification split."""
    # 선택 기능이므로 import는 실제 iterative method를 쓸 때만 수행한다.
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError as exc:
        raise ImportError(
            "iterative split을 사용하려면 iterative-stratification 패키지가 필요합니다. "
            "uv sync 후 다시 실행하세요."
        ) from exc

    # iterstrat 입력 형식에 맞게 파일 목록 X와 multi-hot label 행렬 y를 만든다.
    y = _make_multilabel_matrix(file_names, labels_by_id, image_id_by_file, num_classes)
    x = np.arange(len(file_names)).reshape(-1, 1)

    # test_size가 val_ratio 역할을 하며, random_state로 재현 가능한 분할을 만든다.
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=seed,
    )
    train_idx, val_idx = next(splitter.split(x, y))

    # 인덱스 결과를 Dataset이 받는 파일명 리스트로 되돌린다.
    train_files = [file_names[idx] for idx in train_idx.tolist()]
    val_files = [file_names[idx] for idx in val_idx.tolist()]
    train_files, val_files = _ensure_rare_classes_in_val(
        train_files=train_files,
        val_files=val_files,
        labels_by_id=labels_by_id,
        image_id_by_file=image_id_by_file,
        num_classes=num_classes,
    )
    return train_files, val_files


def _make_multilabel_matrix(
    file_names: list[str],
    labels_by_id: dict[int, set[int]],
    image_id_by_file: dict[str, int],
    num_classes: int,
) -> np.ndarray:
    """image_id -> class_id 집합을 [num_images, num_classes] multi-hot 행렬로 바꾼다."""
    # 행은 이미지, 열은 class_id이며 해당 이미지에 클래스가 있으면 1로 표시한다.
    y = np.zeros((len(file_names), num_classes), dtype=np.int64)

    # annotation이 없는 파일은 전부 0인 행으로 남겨 두어 split 대상에서는 유지한다.
    for row_idx, file_name in enumerate(file_names):
        image_id = image_id_by_file.get(file_name)
        if image_id is None:
            continue

        # class_id 범위를 벗어난 값은 config 매핑 불일치 가능성이 있어 행렬에는 반영하지 않는다.
        for class_id in labels_by_id.get(image_id, set()):
            if 0 <= class_id < num_classes:
                y[row_idx, class_id] = 1

    return y


def _ensure_rare_classes_in_val(
    train_files: list[str],
    val_files: list[str],
    labels_by_id: dict[int, set[int]],
    image_id_by_file: dict[str, int],
    num_classes: int,
) -> tuple[list[str], list[str]]:
    """val에 빠진 희소 class가 있으면 해당 class를 가진 train 이미지를 val로 옮긴다."""
    total_counts = _count_classes(train_files + val_files, labels_by_id, image_id_by_file, num_classes)
    val_counts = _count_classes(val_files, labels_by_id, image_id_by_file, num_classes)

    # 전체 이미지가 2개 이상 있는 class만 val 보정 대상으로 삼는다. 1개뿐인 class는 train에 남기는 편이 안전하다.
    missing_classes = {
        class_id
        for class_id, total_count in enumerate(total_counts)
        if total_count > 1 and val_counts[class_id] == 0
    }

    while missing_classes and len(train_files) > 1:
        candidate = _select_file_covering_missing_classes(
            train_files=train_files,
            missing_classes=missing_classes,
            labels_by_id=labels_by_id,
            image_id_by_file=image_id_by_file,
            total_counts=total_counts,
        )
        if candidate is None:
            break

        train_files.remove(candidate)
        val_files.append(candidate)

        image_id = image_id_by_file.get(candidate)
        candidate_labels = labels_by_id.get(image_id, set()) if image_id is not None else set()
        missing_classes -= candidate_labels

    return train_files, val_files


def _select_file_covering_missing_classes(
    train_files: list[str],
    missing_classes: set[int],
    labels_by_id: dict[int, set[int]],
    image_id_by_file: dict[str, int],
    total_counts: list[int],
) -> str | None:
    """빠진 class를 가장 많이 포함하고, 그중 더 희소한 class를 담은 파일을 고른다."""
    best_file = None
    best_score = None

    for file_name in train_files:
        image_id = image_id_by_file.get(file_name)
        labels = labels_by_id.get(image_id, set()) if image_id is not None else set()
        covered = labels & missing_classes
        if not covered:
            continue

        rarest_count = min(total_counts[class_id] for class_id in covered)
        score = (len(covered), -rarest_count)
        if best_score is None or score > best_score:
            best_file = file_name
            best_score = score

    return best_file


def _count_classes(
    files: list[str],
    labels_by_id: dict[int, set[int]],
    image_id_by_file: dict[str, int],
    num_classes: int,
) -> list[int]:
    """파일 목록에 포함된 class_id별 이미지 수를 센다."""
    counts = [0 for _ in range(num_classes)]
    for file_name in files:
        image_id = image_id_by_file.get(file_name)
        if image_id is None:
            continue
        for class_id in labels_by_id.get(image_id, set()):
            if 0 <= class_id < num_classes:
                counts[class_id] += 1
    return counts


def _save_split_outputs(
    output_dir: Path,
    method: str,
    val_ratio: float,
    seed: int,
    train_files: list[str],
    val_files: list[str],
    random_files: tuple[list[str], list[str]] | None,
    labels_by_id: dict[int, set[int]] | None,
    image_id_by_file: dict[str, int] | None,
    num_classes: int | None,
) -> None:
    """data/splits에 split 파일 목록과 클래스 분포 비교 로그를 저장한다."""
    # 학습 재현을 위해 실제 사용된 train/val 파일 목록을 남긴다.
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_lines(output_dir / "train_files.txt", train_files)
    _write_lines(output_dir / "val_files.txt", val_files)

    # split 방식, seed, 비율, 개수는 한눈에 확인할 수 있게 summary로 저장한다.
    summary_rows = [
        {
            "method": method,
            "seed": seed,
            "val_ratio": val_ratio,
            "num_train": len(train_files),
            "num_val": len(val_files),
        }
    ]
    _write_csv(
        output_dir / "split_summary.csv",
        summary_rows,
        fieldnames=["method", "seed", "val_ratio", "num_train", "num_val"],
    )

    # 라벨 정보가 없으면 파일 목록만 저장하고 클래스 분포 로그는 생략한다.
    if labels_by_id is None or image_id_by_file is None or num_classes is None:
        return

    rows = []

    # iterative일 때만 같은 seed의 random split을 추가 저장해 클래스 분포 비교가 가능하게 한다.
    if random_files is not None:
        random_train, random_val = random_files
        rows.extend(_distribution_rows("random", "train", random_train, labels_by_id, image_id_by_file, num_classes))
        rows.extend(_distribution_rows("random", "val", random_val, labels_by_id, image_id_by_file, num_classes))

    rows.extend(_distribution_rows(method, "train", train_files, labels_by_id, image_id_by_file, num_classes))
    rows.extend(_distribution_rows(method, "val", val_files, labels_by_id, image_id_by_file, num_classes))
    _write_csv(
        output_dir / "split_class_distribution.csv",
        rows,
        fieldnames=["method", "split", "class_id", "count"],
    )


def _distribution_rows(
    method: str,
    split_name: str,
    files: list[str],
    labels_by_id: dict[int, set[int]],
    image_id_by_file: dict[str, int],
    num_classes: int,
) -> list[dict[str, int | str]]:
    # 주어진 파일 목록에 대해 class_id별 등장 이미지 수를 센다.
    counts = _count_classes(files, labels_by_id, image_id_by_file, num_classes)

    # CSV로 저장하기 쉬운 row dict 목록으로 변환한다.
    return [
        {
            "method": method,
            "split": split_name,
            "class_id": class_id,
            "count": count,
        }
        for class_id, count in enumerate(counts)
    ]


def _write_lines(path: Path, lines: list[str]) -> None:
    # train_files.txt / val_files.txt처럼 단순 파일명 목록을 저장한다.
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    # split_summary.csv / split_class_distribution.csv처럼 표 형태 로그를 저장한다.
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
