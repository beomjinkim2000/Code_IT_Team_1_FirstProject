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

    def _load(self) -> dict:
        with open(self.path, "r") as f:
            return yaml.safe_load(f)

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
