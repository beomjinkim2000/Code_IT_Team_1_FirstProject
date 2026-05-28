from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from src.utils.bbox import xywh_to_xyxy


RAW_DATA_ROOT = Path("data/raw/ai11-level1-project/sprint_ai_project1_data")


class PillDataset(Dataset):
    """
    학습 루프가 사용할 수 있는 (image_tensor, target_dict) 형식으로 반환하는 Dataset.

    반환 예시:
        image_tensor: torch.Tensor [C, H, W], float32, 0~1
        target_dict = {
            "boxes": Tensor[N, 4],   # xyxy 절대 픽셀 좌표
            "labels": Tensor[N],     # category_to_label 적용 시 class_id, 기본값은 raw category_id
            "image_id": int,
        }

    category_to_label:
        raw category_id를 모델 학습용 class_id(0~num_classes-1)로 변환한다.
    label_to_category:
        Dataset에서는 사용하지 않고, submission 생성 시 class_id를 raw category_id로 되돌릴 때 사용한다.
    """

    def __init__(
        self,
        root: str | Path = RAW_DATA_ROOT,
        split: str = "train",
        transforms: Callable[[Image.Image, dict[str, Any]], tuple[Any, dict[str, Any]]] | None = None,
        annotations: dict[str, dict[str, Any]] | None = None,
        image_files: list[str] | None = None,
        category_to_label: dict[int, int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.category_to_label = category_to_label

        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        # 현재 원본 데이터는 val 폴더가 따로 없으므로 train/val 모두 train_images를 사용한다.
        image_dir_name = "test_images" if split == "test" else "train_images"
        self.image_dir = self.root / image_dir_name
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # JSON 전체 재스캔을 줄이기 위해, 외부에서 한 번 로드한 annotations를 주입받을 수 있게 했다.
        # 기존: Dataset 생성마다 train_annotations 전체 탐색 -> 현재: annotations 인자 재사용 가능.
        # train/val은 정답 annotation을 사용하고, test는 제출용 이미지라 빈 target으로 처리한다.
        if split == "test":
            self.annotations = {}
        elif annotations is not None:
            self.annotations = annotations
        else:
            self.annotations = self.load_annotations(self.root)

        # split.py에서 만든 파일 목록이 들어오면 해당 subset만 Dataset으로 사용한다.
        self.image_paths = sorted(self.image_dir.glob("*.png"))
        if image_files is not None:
            allowed = set(image_files)
            self.image_paths = [path for path in self.image_paths if path.name in allowed]

        if not self.image_paths:
            if image_files is not None:
                raise FileNotFoundError(
                    f"None of the specified image_files were found in: {self.image_dir}"
                )
            raise FileNotFoundError(f"No PNG images found in: {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx) -> tuple[torch.Tensor, dict]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # 원본 annotation의 xywh bbox를 내부 표준인 xyxy target으로 만든다.
        boxes, labels, image_id = self._get_target(image_path.name, fallback_id=idx)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "original_size": (image.height, image.width),  # transforms 적용 전 원본 크기
        }

        # resize/augmentation처럼 image와 bbox가 함께 바뀌는 작업은 transforms.py에서 처리한다.
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # transforms 이후에도 계약서의 target dtype/shape를 Dataset이 최종 보장한다.
        target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32).reshape(-1, 4)
        target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64).reshape(-1)
        target["image_id"] = int(target["image_id"])

        # Dataset 최종 출력은 [C,H,W] float32 0~1 Tensor여야 한다.
        image_tensor = self._image_to_tensor(image)
        return image_tensor, target

    @classmethod
    def load_annotations(
        cls,
        root: str | Path = RAW_DATA_ROOT,
        corrections_path: str | Path | None = Path("configs/bbox_corrections.json"),
    ) -> dict[str, dict[str, Any]]:
        annotation_dir = Path(root) / "train_annotations"
        if not annotation_dir.exists():
            raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

        # bbox_corrections.json 로드 (없으면 빈 dict → 보정 없이 기존 동작 유지)
        corrections: dict[str, Any] = {}
        if corrections_path is not None:
            cp = Path(corrections_path)
            if cp.exists():
                with open(cp, "r", encoding="utf-8") as f:
                    corrections = json.load(f)

        # 오기입 보정 테이블: {file_name: {원본 bbox tuple → 보정 bbox list}}
        coord_fixes: dict[str, dict[tuple, list]] = {}
        if "bbox_x6567_fix" in corrections:
            fix = corrections["bbox_x6567_fix"]
            coord_fixes[fix["image_file_name"]] = {
                tuple(fix["original_bbox"]): fix["corrected_bbox"]
            }

        # 중복 bbox 제거 대상: {file_name: (중복 bbox tuple, 대체 예측 목록)}
        dup_fix_map: dict[str, tuple[tuple, list]] = {}
        for fix in corrections.get("duplicate_bbox_fixes", []):
            dup_fix_map[fix["image_file_name"]] = (
                tuple(fix["duplicate_bbox_xywh"]),
                fix.get("model_predictions", []),
            )

        annotations_by_file: dict[str, dict[str, Any]] = {}
        for json_path in annotation_dir.rglob("*.json"):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            images_by_id = {int(image["id"]): image for image in data.get("images", [])}

            for annotation in data.get("annotations", []):
                image_id = int(annotation["image_id"])
                image_info = images_by_id[image_id]
                file_name = image_info["file_name"]

                x, y, w, h = annotation["bbox"]
                img_w = image_info.get("width", float("inf"))
                img_h = image_info.get("height", float("inf"))

                # 알려진 오기입 보정 (e.g. bbox_x=6567 → 656)
                if file_name in coord_fixes:
                    x, y, w, h = coord_fixes[file_name].get((x, y, w, h), (x, y, w, h))

                if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > img_w or y + h > img_h:
                    continue

                item = annotations_by_file.setdefault(
                    file_name,
                    {"image_id": image_id, "boxes_xywh": [], "labels": []},
                )
                item["boxes_xywh"].append([x, y, w, h])
                item["labels"].append(int(annotation["category_id"]))

        # 중복 bbox 제거 후 모델 예측으로 대체
        for fname, (dup_bbox, replacements) in dup_fix_map.items():
            if fname not in annotations_by_file:
                continue
            item = annotations_by_file[fname]
            keep = [i for i, b in enumerate(item["boxes_xywh"]) if tuple(b) != dup_bbox]
            item["boxes_xywh"] = [item["boxes_xywh"][i] for i in keep]
            item["labels"] = [item["labels"][i] for i in keep]
            for pred in replacements:
                item["boxes_xywh"].append(pred["bbox_xywh"])
                item["labels"].append(int(pred["category_id"]))

        # annotation 없는 이미지에 모델 예측 추가
        for addition in corrections.get("missing_annotation_additions", []):
            fname = addition["image_file_name"]
            if fname in annotations_by_file:
                continue
            preds = addition.get("model_predictions", [])
            if not preds:
                continue
            annotations_by_file[fname] = {
                "image_id": abs(hash(fname)) % (10 ** 8),
                "boxes_xywh": [p["bbox_xywh"] for p in preds],
                "labels": [int(p["category_id"]) for p in preds],
            }

        return annotations_by_file

    def _get_target(self, file_name: str, fallback_id: int) -> tuple[Tensor, Tensor, int]:
        annotation = self.annotations.get(file_name)
        if annotation is None:
            image_id = self._image_id_from_file_name(file_name, fallback_id)
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64),
                image_id,
            )

        boxes_xywh = torch.tensor(annotation["boxes_xywh"], dtype=torch.float32)
        boxes = xywh_to_xyxy(boxes_xywh)
        raw_labels = [int(label) for label in annotation["labels"]]
        # 학습 시에는 raw category_id를 모델용 class_id(0~num_classes-1)로 바꿔서 사용한다.
        if self.category_to_label is not None:
            labels = torch.tensor(
                [self.category_to_label[label] for label in raw_labels],
                dtype=torch.int64,
            )
        else:
            labels = torch.tensor(raw_labels, dtype=torch.int64)
        return boxes, labels, int(annotation["image_id"])

    @staticmethod
    def _image_to_tensor(image: Any) -> Tensor:
        # 기존: np.asarray(image, dtype=np.float32) / 255.0 수동 변환.
        # 현재: torchvision to_tensor로 [C,H,W] 변환과 0~1 scaling을 표준 함수에 맡긴다.
        return to_tensor(image)

    @staticmethod
    def _image_id_from_file_name(file_name: str, fallback_id: int) -> int:
        stem = Path(file_name).stem
        return int(stem) if stem.isdigit() else fallback_id
