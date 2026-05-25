import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm


def _outputs_to_predictions(output: dict, batch_size: int, score_threshold: float = 0.001) -> list[dict[str, Tensor]]:       #모델의 출력값을 torchmetrics 형식의 예측 결과로 변환하는 함수
    if not isinstance(output, dict):
        raise TypeError("evaluate는 boxes/scores/labels 또는 boxes/scores 형태의 dict 출력을 기대합니다.")

    boxes = output["boxes"].detach().cpu()          #예측 박스 좌표를 CPU로 이동
    scores = output["scores"].detach().cpu()        #예측 점수를 CPU로 이동

    if "labels" in output:      #후처리에서 클래스 ID까지 만든 경우
        conf = scores
        labels = output["labels"].detach().cpu().long()
    else:       #클래스별 점수에서 가장 높은 점수와 클래스 ID를 가져오는 경우
        if scores.ndim != 3:
            raise ValueError("labels가 없으면 scores는 [batch, box개수, num_classes] 형태여야 합니다.")
        conf, labels = scores.max(dim=-1)

    predictions = []        #이미지별 예측 결과를 저장할 리스트
    for idx in range(batch_size):
        keep = conf[idx] > score_threshold      #점수가 기준값보다 높은 예측만 남김
        predictions.append(
            {
                "boxes": boxes[idx][keep],      #남은 예측 박스 좌표
                "scores": conf[idx][keep],      #남은 예측 점수
                "labels": labels[idx][keep].long(),     #남은 예측 클래스 ID(알약 종류 번호)
            }
        )
    return predictions


def evaluate(model: nn.Module, dataloader: DataLoader, device) -> dict:     #모델을 평가하는 함수
    model.eval()        #평가 모드 전환
    metric = MeanAveragePrecision(iou_type="bbox")     #torchmetrics의 bbox mAP 계산 객체 생성

    with torch.no_grad():
        for images, batch_targets in tqdm(dataloader, desc="evaluate", leave=False):        #dataloader에서 이미지와 정답 데이터를 하나씩 가져옴
            batch = torch.stack(images).to(device)      #이미지를 배치 형태로 묶어서 모델이 있는 장치로 이동
            output = model(batch)       #모델에 배치를 입력하여 예측 결과를 얻음
            predictions = _outputs_to_predictions(output, batch.shape[0])       #torchmetrics 형식의 예측 결과로 변환
            targets = [     #torchmetrics 형식의 정답 데이터로 변환
                {
                    "boxes": target["boxes"].detach().cpu(),        #정답 박스 좌표를 CPU로 이동
                    "labels": target["labels"].detach().cpu().long(),       #정답 라벨을 CPU로 이동하고 int64 정수 타입으로 변환
                }
                for target in batch_targets     #정답데이터에서 하나씩 봄
            ]
            metric.update(predictions, targets)     #현재 batch의 예측과 정답을 mAP 계산기에 추가

    result = metric.compute()       #전체 validation 결과에 대한 mAP 계산
    return {"mAP": float(result["map"]), "mAP_50": float(result["map_50"])}      #mAP와 IoU 0.5 기준 mAP 반환
