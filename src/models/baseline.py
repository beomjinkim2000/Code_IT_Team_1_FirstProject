import torch
from torch import nn
from ultralytics import YOLO

from src.utils.config import load_config


def _adapt_num_classes(model: nn.Module, num_classes: int) -> None:
    # YOLO 프리트레인 Detect Head를 알약 클래스 수에 맞게 교체

    detect_head = model.model[-1]  # 마지막 레이어 = Detect Head
    detect_head.nc = num_classes
    detect_head.no = num_classes + detect_head.reg_max * 4  # reg_max * 4: 4좌표(l/t/r/b) 각각 reg_max개 DFL 분포

    for branch in detect_head.cv3:  # 클래스 예측 브랜치 3개 (P3/P4/P5)
        last_layer = branch[-1]
        new_layer = nn.Conv2d(
            out_channels=num_classes,
            in_channels=last_layer.in_channels,
            kernel_size=last_layer.kernel_size,
            stride=last_layer.stride,
            padding=last_layer.padding,
            bias=last_layer.bias is not None,
        )
        nn.init.kaiming_normal_(new_layer.weight, mode="fan_in", nonlinearity="relu")
        if new_layer.bias is not None:
            nn.init.zeros_(new_layer.bias)
        branch[-1] = new_layer

    model.nc = num_classes
    model.names = {i: str(i) for i in range(num_classes)}


def build_model(num_classes: int) -> torch.nn.Module:
    cfg = load_config(validate=False)
    model_name = cfg["model"]["name"]

    yolo = YOLO(f"{model_name}.pt")
    model = yolo.model

    _adapt_num_classes(model, num_classes)
    return model
