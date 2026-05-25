from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(        #모델과 옵티마이저의 상태를 저장하는 함수
    model: nn.Module,       #저장할 모델
    optimizer: Optimizer,   #저장할 옵티마이저
    epoch: int,             #현재 epoch 번호
    val_mAP: float,         #현재 validation mAP 점수
    checkpoint_dir: str | Path = "outputs/checkpoints",     #체크포인트를 저장할 디렉토리 경로, 기본값은 "outputs/checkpoints"
    is_best: bool = False,      #현재 모델이 지금까지 저장된 모델 중 가장 좋은 모델인지 여부, 기본값은 False
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)       #checkpoint_dir 경로를 Path 객체로 변환
    checkpoint_dir.mkdir(parents=True, exist_ok=True)       #checkpoint_dir이 존재하지 않으면 디렉토리를 생성, parents=True는 상위 디렉토리도 함께 생성, exist_ok=True는 이미 디렉토리가 존재해도 에러 없이 넘어가도록 함

    checkpoint = {      #체크포인트에 저장할 정보들을 딕셔너리 형태로 만듦
        "epoch": epoch,     #현재 epoch 번호
        "model_state": model.state_dict(),      #모델의 상태 사전 (모델의 가중치와 버퍼 정보)
        "optimizer_state": optimizer.state_dict(),      #옵티마이저의 상태 사전 (옵티마이저의 매개변수와 내부 상태 정보)
        "val_mAP": val_mAP,     #현재 validation mAP 점수
    }

    path = checkpoint_dir / f"epoch_{epoch}.pt"     #체크포인트 파일의 경로를 epoch 번호를 포함하여 만듦 ex) "outputs/checkpoints/epoch_10.pt"
    torch.save(checkpoint, path)        #체크포인트 딕셔너리를 지정한 경로에 저장

    if is_best:     #현재 모델이 가장 좋은 모델인 경우, "best_model.pt"라는 이름으로 체크포인트를 저장하여 나중에 쉽게 불러올 수 있도록 함
        torch.save(checkpoint, checkpoint_dir / "best_model.pt")        #checkpoint 딕셔너리를 "best_model.pt"라는 이름으로 저장

    return path


def load_checkpoint(        #저장된 체크포인트 파일에서 모델과 옵티마이저의 상태를 불러오는 함수
    path: str | Path,       #불러올 체크포인트 파일의 경로
    model: nn.Module,       #체크포인트에서 불러온 모델의 상태를 저장할 모델 객체
    optimizer: Optimizer | None = None,     #체크포인트에서 불러온 옵티마이저의 상태를 저장할 옵티마이저 객체, 기본값은 None (옵티마이저 상태를 불러오지 않음)
    device: str | torch.device = "cpu",     #체크포인트를 불러올 때 모델과 옵티마이저의 상태를 이동시킬 장치, 기본값은 "cpu"
) -> dict:      #체크포인트에서 불러온 정보들을 딕셔너리 형태로 반환하는 함수
    checkpoint = torch.load(path, map_location=device)      #체크포인트 파일을 지정한 장치로 불러옴
    model.load_state_dict(checkpoint["model_state"])        #체크포인트에서 불러온 모델의 상태를 모델 객체에 로드

    if optimizer is not None:       #옵티마이저가 제공된 경우
        optimizer.load_state_dict(checkpoint["optimizer_state"])        #체크포인트에서 불러온 옵티마이저의 상태를 옵티마이저 객체에 로드

    return checkpoint
