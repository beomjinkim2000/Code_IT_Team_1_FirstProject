import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou


def compute_per_class_f1(
    predictions: list[dict],
    targets: list[dict],
    num_classes: int,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
) -> dict[int, dict]:
    """val set에서 클래스별 precision / recall / F1을 계산한다.

    confidence >= conf_threshold인 예측만 사용.
    예측 박스와 GT 박스의 IoU >= iou_threshold + 클래스 일치 시 TP.

    반환: {class_id: {"precision": float, "recall": float, "f1": float}}
    클래스에 GT도 예측도 없으면 해당 class_id는 결과에 포함되지 않는다.
    """
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)

    for pred, tgt in zip(predictions, targets):
        pred_boxes = pred["boxes"].cpu().float()
        pred_scores = pred["scores"].cpu().float()
        pred_labels = pred["labels"].cpu().long()
        gt_boxes = tgt["boxes"].cpu().float()
        gt_labels = tgt["labels"].cpu().long()

        keep = pred_scores >= conf_threshold
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

        order = pred_scores.argsort(descending=True)
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]

        for pb, pl in zip(pred_boxes, pred_labels):
            cls = int(pl)
            same_cls_idx = ((gt_labels == cls) & ~gt_matched).nonzero(as_tuple=True)[0]
            if len(same_cls_idx) == 0:
                fp[cls] += 1
                continue
            ious = box_iou(pb.unsqueeze(0), gt_boxes[same_cls_idx])[0]
            best = ious.argmax()
            if ious[best] >= iou_threshold:
                tp[cls] += 1
                gt_matched[same_cls_idx[best]] = True
            else:
                fp[cls] += 1

        for gl in gt_labels[~gt_matched]:
            fn[int(gl)] += 1

    result = {}
    for c in range(num_classes):
        if tp[c] + fp[c] + fn[c] == 0:
            continue
        p = float(tp[c] / (tp[c] + fp[c] + 1e-9))
        r = float(tp[c] / (tp[c] + fn[c] + 1e-9))
        f1 = 2 * p * r / (p + r + 1e-9)
        result[c] = {"precision": round(p, 6), "recall": round(r, 6), "f1": round(f1, 6)}

    return result


def evaluate(predictions: list[dict], targets: list[dict]) -> dict:
    """postprocess 출력(list[Prediction])과 정답 dict 리스트를 받아 mAP를 계산한다.

    predictions: postprocess_raw_outputs() 결과 — boxes/scores/labels 키 포함
    targets:     DataLoader에서 온 target dict — boxes/labels 키 포함
    """
    metric = MeanAveragePrecision(iou_type="bbox")

    preds_fmt = [
        {
            "boxes":  p["boxes"].detach().cpu().float(),
            "scores": p["scores"].detach().cpu().float(),
            "labels": p["labels"].detach().cpu().long(),
        }
        for p in predictions
    ]
    tgts_fmt = [
        {
            "boxes":  t["boxes"].detach().cpu().float(),
            "labels": t["labels"].detach().cpu().long(),
        }
        for t in targets
    ]

    metric.update(preds_fmt, tgts_fmt)
    result = metric.compute()
    return {"mAP": float(result["map"]), "mAP_50": float(result["map_50"])}
