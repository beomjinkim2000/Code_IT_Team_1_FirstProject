import torch
from torch import nn
from ultralytics import YOLO

# YOLO모델 기존에 학습된 클래스를 필요한 클래스(알약) 수에 맞게 변경하는 함수
def _adapt_num_classes(model: nn.Module, num_classes: int) -> None:
    if num_classes is None or num_classes <= 0:     #num_classes가 None 이거나 0이하인 경우
        raise ValueError(f"num_classes는 1 이상이어야 합니다. 현재 값: {num_classes}")

    detect_head = model.model[-1]       #YOLO 모델 마지막에 있는 Detect Head를 가져옴
    detect_head.nc = num_classes        #Detect Head의 클래스 수를 num_classes로 변경  (config.yaml의 num_classes와 일치)
    detect_head.no = num_classes + detect_head.reg_max * 4      #num_classes + 4개의 bbox 좌표 예측값(left, top, right, bottom) * 4

    #각 예측 레이어의 마지막 레이어를 num_classes에 맞게 변경
    for branch in detect_head.cv3:      #class 예측을 담당하는 레이어 묶음들
        last_layer = branch[-1]     #마지막 레이어 를 가져옴
        branch[-1] = nn.Conv2d(
            out_channels=num_classes,       #출력 채널만 num_classes로 변경
            in_channels=last_layer.in_channels,     #기존 설정 동일
            kernel_size=last_layer.kernel_size,
            stride=last_layer.stride,
            padding=last_layer.padding,
            bias=last_layer.bias is not None,
        )

    model.nc = num_classes      #모델에 num_classes 정보를 저장
    model.names = {i: str(i) for i in range(num_classes)}       #현재 클래스 이름을 알 수 없으니 임시로 저장, 추후 train.py에서 dataset의 class_names로 덮어씌울 예정


def build_model(num_classes: int) -> torch.nn.Module:
    yolo = YOLO("yolov8n.pt")
    model = yolo.model

    _adapt_num_classes(model, num_classes)
    model.train()       #train모드로 설정

    return model
