# 인터페이스 계약서

> **이 파일이 바뀌면 팀장 승인 필요.**
> 병렬 개발 중 형식이 달라지면 merge 시 코드가 깨진다.

---

## 1. Dataset 출력 형식

```python
image, target = dataset[idx]

# image: torch.Tensor [C, H, W], float32, 0~1 정규화
target = {
    "boxes":    torch.Tensor,   # shape [N, 4], dtype float32
                                # 형식: [x1, y1, x2, y2] 절대 픽셀 좌표
    "labels":   torch.Tensor,   # shape [N], dtype int64, category_id
    "image_id": int
}
```

---

## 2. bbox 내부 형식 규칙

| 용도 | 형식 | 위치 |
|---|---|---|
| 학습 / 예측 내부 | `[x1, y1, x2, y2]` 절대 픽셀 | 모든 내부 코드 |
| Kaggle 제출 CSV | `[x, y, w, h]` 좌상단 기준 | `make_submission.py` 출력 |
| 변환 함수 | `xyxy_to_xywh()` / `xywh_to_xyxy()` | `src/utils/bbox.py` |

---

## 3. 모델 생성 함수

```python
# src/models/baseline.py
def build_model(num_classes: int) -> torch.nn.Module:
    ...
```

---

## 4. predict 결과 형식

```python
# src/engine/predict.py 반환값: List[Dict]
pred = {
    "image_id": int,
    "boxes":    torch.Tensor,   # [N, 4], [x1, y1, x2, y2]
    "labels":   torch.Tensor,   # [N], int64
    "scores":   torch.Tensor,   # [N], float32
}
```

---

## 5. submission.csv 컬럼

```
annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score
```

- `bbox_x, bbox_y, bbox_w, bbox_h`: xywh 형식 (절대 픽셀)
- `annotation_id`: 전체 예측 결과 순번 (1부터 시작, unique)

---

## 변경 이력

| 날짜 | 변경 내용 | 승인자 |
|---|---|---|
| 2026-05-19 | 초안 작성 | 김범진 |
