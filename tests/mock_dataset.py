"""
validate_dataset 테스트용 Mock 데이터셋.

실제 데이터 없이 두 가지 용도로 사용:
  1. 자기 dataset.py가 올바른 형식을 반환하는지 확인할 때 기준으로 삼기
  2. validate_dataset 자체가 각 조건을 제대로 잡는지 확인

실행:
  python tests/mock_dataset.py
"""

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import Dataset

from src.utils.validate import validate_dataset


class MockDataset(Dataset):
    """
    interfaces.md 스펙을 만족하는 가짜 데이터셋.

    break_rule로 특정 조건을 의도적으로 위반한 샘플 생성 가능.
    기본값(None)은 모든 조건을 통과하는 올바른 형식.
    """

    BREAK_RULES = [
        "not_tensor",     # torch.Tensor가 아님
        "wrong_ndim",     # 차원이 [C,H,W]가 아님
        "wrong_channel",  # 채널 수가 3이 아님
        "wrong_size",     # H, W가 32배수 아님
        "wrong_dtype",    # dtype이 float32 아님  (warning)
        "wrong_range",    # 값 범위가 0~1 벗어남  (warning)
    ]

    def __init__(
        self,
        size: int = 20,
        img_size: int = 640,
        num_classes: int = 10,
        break_rule: str = None,
    ):
        assert break_rule in (self.BREAK_RULES + [None]), \
            f"break_rule은 {self.BREAK_RULES} 중 하나여야 함"
        self.size = size
        self.img_size = img_size
        self.num_classes = num_classes
        self.break_rule = break_rule

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx):
        return self._make_image(), self._make_target(idx)

    # ── private ──────────────────────────────────────────────────────────────

    def _make_image(self):
        base = torch.rand(3, self.img_size, self.img_size)  # 정상: float32, 0~1

        if self.break_rule == "not_tensor":
            return base.numpy()                             # ndarray 반환
        if self.break_rule == "wrong_ndim":
            return base.unsqueeze(0).unsqueeze(0)           # ndim=5
        if self.break_rule == "wrong_channel":
            return base[:1]                                 # 채널=1
        if self.break_rule == "wrong_size":
            return torch.rand(3, 100, 100)                  # 32배수 아님
        if self.break_rule == "wrong_dtype":
            return (base * 255).to(torch.uint8)             # uint8
        if self.break_rule == "wrong_range":
            return base * 255                               # 0~255

        return base

    def _make_target(self, idx: int) -> dict:
        n = random.randint(0, 4)  # 알약 0~4개 (0개 케이스도 포함)

        if n == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)
        else:
            x1 = torch.rand(n) * (self.img_size - 60)
            y1 = torch.rand(n) * (self.img_size - 60)
            x2 = (x1 + torch.rand(n) * 50 + 10).clamp(max=self.img_size)
            y2 = (y1 + torch.rand(n) * 50 + 10).clamp(max=self.img_size)
            boxes  = torch.stack([x1, y1, x2, y2], dim=1)
            labels = torch.randint(0, self.num_classes, (n,), dtype=torch.int64)

        return {"boxes": boxes, "labels": labels, "image_id": idx}


# ── 테스트 실행 ───────────────────────────────────────────────────────────────

def run(img_size: int = 640) -> None:
    print("=" * 55)
    print(" validate_dataset Mock 테스트")
    print("=" * 55)

    # ── 정상 케이스 ───────────────────────────────────────────
    print("\n[1] 정상 형식 — 모든 조건 통과해야 함")
    validate_dataset(MockDataset(img_size=img_size), img_size)

    # ── 필수 조건 위반 (AssertionError 예상) ──────────────────
    hard_rules = [
        ("not_tensor",    "torch.Tensor가 아닌 경우"),
        ("wrong_ndim",    "차원이 잘못된 경우"),
        ("wrong_channel", "채널 수가 3이 아닌 경우"),
        ("wrong_size",    "H, W가 32배수 아닌 경우"),
    ]

    print("\n[2] 필수 조건 위반 — AssertionError가 발생해야 정상")
    for rule, desc in hard_rules:
        ds = MockDataset(img_size=img_size, break_rule=rule)
        try:
            validate_dataset(ds, img_size)
            print(f"  ❌ FAIL  {desc} → 에러가 나야 하는데 통과됨")
        except AssertionError:
            print(f"  ✅ PASS  {desc}")

    # ── 권장 조건 위반 (warning만, 통과 예상) ─────────────────
    soft_rules = [
        ("wrong_dtype",  "dtype이 float32 아닌 경우 (warning)"),
        ("wrong_range",  "값 범위가 0~1 벗어난 경우 (warning)"),
    ]

    print("\n[3] 권장 조건 위반 — warning 출력 후 통과해야 정상")
    for rule, desc in soft_rules:
        ds = MockDataset(img_size=img_size, break_rule=rule)
        try:
            validate_dataset(ds, img_size)
            print(f"  ✅ PASS  {desc}")
        except AssertionError:
            print(f"  ❌ FAIL  {desc} → warning이어야 하는데 에러 발생")

    print("\n" + "=" * 55)
    print(" 테스트 완료")
    print("=" * 55)


if __name__ == "__main__":
    run()
