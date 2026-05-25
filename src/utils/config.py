import json
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "default.yaml"


def _get_nested(cfg: dict, dotted_key: str) -> Any:
    """'model.num_classes' 같은 점 표기 키로 중첩 dict 값 반환."""
    node = cfg
    for k in dotted_key.split("."):
        if not isinstance(node, dict) or k not in node:
            raise KeyError(f"'{dotted_key}' 키를 찾을 수 없음")
        node = node[k]
    return node


class ConfigLoader:
    """
    default.yaml을 로드하고 _required 필드의 null 여부를 검사.

    _required 목록은 yaml 파일 안에 선언:
        _required:
          - model.num_classes
          - data.nc

    사용:
        loader = ConfigLoader()
        loader.validate_required()   # null 있으면 ValueError
        cfg = loader.cfg
    """

    def __init__(self, path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self.path = Path(path)
        self.cfg: dict = self._load()
        self._load_class_mapping()

    def _load(self) -> dict:
        with open(self.path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_class_mapping(self) -> None:
        mapping_path = self.cfg.get("data", {}).get("class_mapping")
        if not mapping_path:
            return

        path = Path(mapping_path)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path

        with open(path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        classes = mapping.get("classes", [])
        if not classes:
            raise ValueError(f"class_mapping 파일에 classes가 없습니다: {path}")

        classes = sorted(classes, key=lambda item: int(item["class_id"]))
        class_ids = [int(item["class_id"]) for item in classes]
        expected_ids = list(range(len(classes)))
        if class_ids != expected_ids:
            raise ValueError("class_mapping의 class_id는 0부터 연속된 정수여야 합니다.")

        category_ids = [int(item["category_id"]) for item in classes]
        if len(category_ids) != len(set(category_ids)):
            raise ValueError("class_mapping에 중복 category_id가 있습니다.")

        num_classes = int(mapping.get("num_classes", len(classes)))
        if num_classes != len(classes):
            raise ValueError(
                f"class_mapping num_classes({num_classes})와 classes 개수({len(classes)})가 다릅니다."
            )

        data_cfg = self.cfg.setdefault("data", {})
        model_cfg = self.cfg.setdefault("model", {})
        data_cfg["nc"] = num_classes
        data_cfg["names"] = [str(item["name"]) for item in classes]
        data_cfg["category_to_label"] = {
            int(item["category_id"]): int(item["class_id"]) for item in classes
        }
        data_cfg["label_to_category"] = {
            int(item["class_id"]): int(item["category_id"]) for item in classes
        }
        model_cfg["num_classes"] = num_classes

    def validate_required(self) -> None:
        """_required에 등록된 필드 중 null / 빈 값인 항목을 모두 보고."""
        required_fields: list[str] = self.cfg.get("_required", [])
        if not required_fields:
            return

        unfilled: list[str] = []
        for field in required_fields:
            try:
                val = _get_nested(self.cfg, field)
            except KeyError:
                unfilled.append(f"{field}  ← 키 없음")
                continue
            if val is None or val == [] or val == "":
                unfilled.append(field)

        if unfilled:
            raise ValueError(
                "configs/default.yaml에 아직 채워지지 않은 필수 필드:\n"
                + "\n".join(f"  - {f}" for f in unfilled)
                + "\n  → EDA 완료 후 값을 입력하고 재실행하세요."
            )


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    validate: bool = True,
) -> dict:
    """
    default.yaml을 로드.

    validate=True(기본값)이면 _required 필드 null 검증도 함께 실행.
    EDA 전 탐색 목적이라면 validate=False로 호출.
    """
    loader = ConfigLoader(path)
    if validate:
        loader.validate_required()
    return loader.cfg
