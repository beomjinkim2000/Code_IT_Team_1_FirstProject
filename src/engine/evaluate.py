from torchmetrics.detection.mean_ap import MeanAveragePrecision


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
