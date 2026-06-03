from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Any

from src.data.dataset import PillDataset

WANDB_ENTITY = "health-eat-pill-detection"
WANDB_PROJECT = "health-eat-pill-detection"


def _artifact_name(img_size: int) -> str:
    return f"{WANDB_ENTITY}/{WANDB_PROJECT}/synth-dataset-{img_size}:latest"


def _download_from_wandb(synth_root: Path, img_size: int = 640) -> None:
    import wandb

    api = wandb.Api()
    artifact = api.artifact(_artifact_name(img_size))
    artifact.download(root=str(synth_root.parent))
    print(f"WandB artifact 다운로드 완료: {synth_root}")


class SynthPillDataset(PillDataset):
    """
    build_synth_dataset.py 로 생성된 합성 데이터셋 로더.

    디렉토리 구조:
        {synth_root}/
            images/          ← 합성 PNG
            annotations.json ← COCO 형식

    synth_root 가 없고 WANDB_API_KEY 가 있으면 WandB artifact 에서 자동 다운로드.
    → Colab 런타임 재시작 후에도 최초 1회 빌드 결과를 재사용할 수 있다.

    사용 예시:
        from torch.utils.data import ConcatDataset
        synth_ds = SynthPillDataset(
            synth_root="data/augmented/synth",
            transforms=train_transform(img_size, aug_cfg),
            category_to_label=category_to_label,
        )
        combined = ConcatDataset([train_ds, synth_ds])
    """

    def __init__(
        self,
        synth_root: str | Path,
        transforms: Callable | None = None,
        category_to_label: dict[int, int] | None = None,
        img_size: int = 640,
    ) -> None:
        self.synth_root = Path(synth_root)
        self.split = "train"
        self.transforms = transforms
        self.category_to_label = category_to_label

        self.image_dir = self.synth_root / "images"

        # 로컬에 없으면 WandB artifact에서 자동 다운로드
        if not self.image_dir.exists():
            if os.environ.get("WANDB_API_KEY"):
                print(f"합성 데이터 없음 → WandB artifact 다운로드 중... (synth-dataset-{img_size})")
                _download_from_wandb(self.synth_root, img_size=img_size)
            else:
                raise FileNotFoundError(
                    f"합성 이미지 폴더 없음: {self.image_dir}\n"
                    "build_synth_dataset.py 를 먼저 실행하거나 WANDB_API_KEY 를 설정하세요."
                )

        self.annotations = self._load_synth_annotations(self.synth_root / "annotations.json")
        self.image_paths = sorted(self.image_dir.glob("*.png"))

        if not self.image_paths:
            raise FileNotFoundError(f"PNG 없음: {self.image_dir}")

    @staticmethod
    def _load_synth_annotations(ann_path: Path) -> dict[str, dict[str, Any]]:
        """
        COCO JSON → PillDataset.annotations 와 동일한 dict 형식으로 변환.
        {file_name: {"image_id": int, "boxes_xywh": [[x,y,w,h], ...], "labels": [cat_id, ...]}}
        """
        if not ann_path.exists():
            raise FileNotFoundError(f"어노테이션 파일 없음: {ann_path}")

        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        images_by_id = {img["id"]: img["file_name"] for img in coco["images"]}
        result: dict[str, dict[str, Any]] = {}

        for ann in coco["annotations"]:
            image_id = ann["image_id"]
            fname = images_by_id[image_id]
            item = result.setdefault(
                fname,
                {"image_id": image_id, "boxes_xywh": [], "labels": []},
            )
            item["boxes_xywh"].append(ann["bbox"])        # [x, y, w, h]
            item["labels"].append(int(ann["category_id"]))

        return result
