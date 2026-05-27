import torch
from torch import nn
from ultralytics import YOLO
from src.utils.config import load_config


# YOLO모델 기존에 학습된 클래스를 필요한 클래스(알약) 수에 맞게 변경하는 함수
def _adapt_num_classes(model: nn.Module, num_classes: int) -> None:

    detect_head = model.model[-1]       #YOLO 모델 마지막에 있는 Detect Head를 가져옴
    detect_head.nc = num_classes        #Detect Head의 클래스 수를 num_classes로 변경  (config.yaml의 num_classes와 일치)
    detect_head.no = num_classes + detect_head.reg_max * 4      #num_classes + 4개의 bbox 좌표 예측값(left, top, right, bottom) * 4

    #각 예측 레이어의 마지막 레이어를 num_classes에 맞게 변경
    for branch in detect_head.cv3:      #class 예측을 담당하는 레이어 묶음들
        last_layer = branch[-1]     #마지막 레이어 를 가져옴
        new_layer = nn.Conv2d(
            out_channels=num_classes,       #출력 채널만 num_classes로 변경
            in_channels=last_layer.in_channels,     #기존 설정 동일
            kernel_size=last_layer.kernel_size,
            stride=last_layer.stride,
            padding=last_layer.padding,
            bias=last_layer.bias is not None,
        )
        nn.init.kaiming_normal_(new_layer.weight, mode="fan_in", nonlinearity="relu")       #채널 입력수 에 맞춰(fan_in) weight 값 조정
        if new_layer.bias is not None:      #bias가 있는 경우
            nn.init.zeros_(new_layer.bias)      #bias는 0으로 초기화
        branch[-1] = new_layer      #new_layer에 마지막 레이어에 변경한 weight, bias값을 저장

    model.nc = num_classes      #모델에 num_classes 정보를 저장
    model.names = {i: str(i) for i in range(num_classes)}
    #model.names = {i: name for i, name in enumerate(class_names)}      #class_names가 작성되면 해당 코드로 수정


def freeze_except_cv3_last(model: nn.Module) -> None:
    """cv3 마지막 Conv2d(새로 초기화된 분류 레이어)만 남기고 전체 고정."""
    for param in model.parameters():
        param.requires_grad = False
    for branch in model.model[-1].cv3:
        for param in branch[-1].parameters():
            param.requires_grad = True


def set_frozen_bn_eval(model: nn.Module) -> None:
    """requires_grad=False인 BatchNorm을 eval 모드로 고정.

    model.train() 호출 후 반드시 한 번 더 불러야 한다.
    model.train()은 모든 서브모듈을 train 모드로 리셋하므로
    frozen BN이 배치 통계를 쓰는 버그가 생긴다.
    Phase 3처럼 전체 unfreeze 상태에서는 no-op.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def unfreeze_head(model: nn.Module) -> None:
    """Detect head(layer 22) 전체를 학습 가능 상태로 전환."""
    for param in model.model[-1].parameters():
        param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """전체 레이어 가중치를 학습 가능 상태로 되돌린다."""
    for param in model.parameters():
        param.requires_grad = True


def build_model(num_classes: int) -> torch.nn.Module:
    cfg = load_config(validate=False)
    model_name = cfg["model"]["name"]

    yolo = YOLO(f"{model_name}.pt")
    model = yolo.model

    _adapt_num_classes(model, num_classes)
    model.model[-1].bias_init()
    return model
