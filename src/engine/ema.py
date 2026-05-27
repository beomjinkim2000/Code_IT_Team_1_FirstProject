from __future__ import annotations

import copy

import torch
from torch import nn

from src.utils.config import load_config


def _load_ema_decay() -> float:
    cfg = load_config()     #default.yaml 파일을 로드하여 cfg 변수에 저장
    return float(cfg["ema"]["decay"])       #default.yaml 파일에서 ema decay 값을 가져와서 float로 변환하여 반환


class ModelEMA:
    def __init__(self, model: nn.Module) -> None:
        self.decay = _load_ema_decay()      #default의 decay의 값을 가져옴
        self.model = copy.deepcopy(model).eval()        #현재 모델을 복사해서 ema 전용 모델을 만듦
        for param in self.model.parameters():
            param.requires_grad_(False)     #EMA 모델의 파라미터는 학습되지 않도록 설정

    @torch.no_grad()        #가중치 계산을 하지 않도록 설정
    def update(self, model: nn.Module) -> None:
        ema_state = self.model.state_dict()     #ema 모델의 state dict를 가져옴
        model_state = model.state_dict()        #현재 모델의 state dict를 가져옴

        for key, ema_value in ema_state.items():        #ema 모델의 각 파라미터 ex) weight, bias 등
            model_value = model_state[key].detach()     #key에 해당하는 모델의 값을 가져옴
            if torch.is_floating_point(ema_value):      #ema 모델의 값이 소수인지 라면
                updated_value = self.decay * ema_value + (1.0 - self.decay) * model_value       #decay * ema_value + (1 - decay) * model_value
                ema_value.copy_(updated_value)      #업데이트된 값을 ema 모델의 해당 파라미터에 복사
            else:       #소수형이 아니라면(정수형)
                ema_value.copy_(model_value)        #ema 모델에 현재 모델의 값을 복사

    def state_dict(self) -> dict:       #checkpoint 저장읋 위해 EMA 모델의 state dict를 반환
        return self.model.state_dict()      #EMA 모델의 state dict를 반환하여 EMA 모델의 가중치를 가져옴

    def load_state_dict(self, state_dict: dict) -> None:        #checkpoint에서 EMA 모델의 state dict를 로드하여 EMA 모델의 가중치를 업데이트
        self.model.load_state_dict(state_dict)      #EMA 모델의 state dict를 로드하여 EMA 모델의 가중치를 업데이트

    def to(self, device: torch.device | str) -> "ModelEMA":
        self.model.to(device)       #EMA 모델을 지정된 장치로 이동
        return self

