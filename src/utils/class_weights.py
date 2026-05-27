from __future__ import annotations

from pathlib import Path


def compute_sample_weights(
    image_paths: list,
    annotations: dict,
    category_to_label: dict[int, int],
    num_classes: int,
    method: str = "inverse_freq",
    manual: dict[int, float] | None = None,
) -> list[float]:
    """각 학습 이미지의 샘플링 가중치를 반환한다.

    WeightedRandomSampler에 바로 넘길 수 있는 float 리스트.

    method="inverse_freq": class_weight[c] = max_count / count_c
        가장 많이 등장한 클래스 ≈ 1, 희귀 클래스는 비율만큼 증폭.
        이미지 하나에 여러 클래스가 있으면 그 중 가장 높은 weight를 사용.
    method="manual": manual dict의 {class_id: weight}를 class_weight로 직접 사용.
        지정되지 않은 클래스는 1.0.
    """
    if method == "manual":
        class_weight = [1.0] * num_classes
        if manual:
            for class_id, w in manual.items():
                class_weight[int(class_id)] = float(w)
    else:
        counts = [0] * num_classes
        for ann in annotations.values():
            for raw_cat_id in ann["labels"]:
                label = category_to_label.get(int(raw_cat_id))
                if label is not None:
                    counts[label] += 1

        max_count = max(counts) if any(counts) else 1
        class_weight = [max_count / max(c, 1) for c in counts]

    weights = []
    for img_path in image_paths:
        file_name = img_path.name if isinstance(img_path, Path) else Path(img_path).name
        ann = annotations.get(file_name)
        if ann is None:
            weights.append(1.0)
            continue

        w = max(
            class_weight[category_to_label[int(c)]]
            for c in ann["labels"]
            if int(c) in category_to_label
        )
        weights.append(w)

    return weights
