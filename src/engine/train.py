from types import SimpleNamespace

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics.utils.loss import v8DetectionLoss


def _prepare_loss_args(
    model: nn.Module,
) -> None:  # YOLO 모델의 손실 계산에 필요한 인자들을 모델에 설정하는 함수
    args = getattr(
        model, "args", {}
    )  # 모델에 args 속성이 있는지 확인하고, 없으면 빈 딕셔너리를 반환
    if isinstance(args, dict):  # args가  dict인 경우
        args = SimpleNamespace(**args)  # dict를 SimpleNamespace로 변환

    for name, value in {
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
    }.items():  # 손실 계산에 필요한 인자들을 딕셔너리로 정의
        if not hasattr(args, name):  # args에 해당 인자가 없는 경우에만 설정
            setattr(args, name, value)  # args에 인자 이름과 값을 설정

    model.args = args


def _targets_to_yolo_batch(
    targets: list[dict], images: list[torch.Tensor], device
) -> dict:  # YOLO 모델에 맞게 타겟 데이터를 변환하는 함수
    batch_idx_list = []
    cls_list = []
    bboxes_list = []

    for image_idx, (image, target) in enumerate(
        zip(images, targets)
    ):  # 이미지와 타겟을 순회하면서 처리
        boxes = target["boxes"].to(
            device
        )  # 타겟에서 박스 좌표를 가져와서 device로 이동
        labels = target["labels"].to(
            device
        )  # 타겟에서 클래스 레이블을 가져와서 device로 이동

        if boxes.numel() == 0:  # 박스가 없는 경우 건너뜀
            continue

        _, height, width = image.shape  # 이미지의 높이와 너비를 가져옴
        x1, y1, x2, y2 = boxes.unbind(dim=1)  # 박스 좌표를 x1, y1, x2, y2로 분리

        # x y w h 변환 및 정규화
        cx = (
            (x1 + x2) / 2
        ) / width  # x중심을 구하기 위해 x1, x2를 더해서 2로 나누고 정규화함
        cy = (
            (y1 + y2) / 2
        ) / height  # y중심을 구하기 위해 y1, y2를 더해서 2로 나누고 정규화함
        w = (x2 - x1) / width  # 너비를 구하기 위해 x2에서 x1을 빼고 너비로 나눔
        h = (y2 - y1) / height  # 높이를 구하기 위해 y2에서 y1을 빼고 높이로 나눔

        batch_idx = torch.full(  # 배치 인덱스 생성
            (boxes.shape[0],),  # 박스의 수만큼 배치 인덱스를 생성
            image_idx,  # 현재 이미지의 인덱스로 채움
            dtype=torch.float32,  # 배치 인덱스는 float32 타입으로 생성
            device=device,  # device로 이동
        )

        batch_idx_list.append(batch_idx)  # 배치 인덱스를 리스트에 추가
        cls_list.append(labels.float())  # 클래스 레이블을 리스트에 추가
        bboxes_list.append(
            torch.stack([cx, cy, w, h], dim=1)
        )  # 박스 좌표를 리스트에 추가 (cx, cy, w, h 형태로 스택)

    if (
        not bboxes_list
    ):  # 박스가 하나도 없는 경우, 빈 텐서를 반환하여 모델이 처리할 수 있도록 함
        return {
            "batch_idx": torch.zeros(
                (0,), dtype=torch.float32, device=device
            ),  # 빈 배치 인덱스 텐서
            "cls": torch.zeros(
                (0,), dtype=torch.float32, device=device
            ),  # 빈 클래스 레이블 텐서
            "bboxes": torch.zeros(
                (0, 4), dtype=torch.float32, device=device
            ),  # 빈 박스 좌표 텐서 (4는 cx, cy, w, h)
        }

    return {
        "batch_idx": torch.cat(
            batch_idx_list
        ),  # 배치 인덱스 리스트를 하나의 텐서로 연결
        "cls": torch.cat(cls_list),  # 클래스 레이블 리스트를 하나의 텐서로 연결
        "bboxes": torch.cat(bboxes_list),  # 박스 좌표 리스트를 하나의 텐서로 연결
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: v8DetectionLoss,
    device,
    ema,
) -> tuple[float, float, float, float]:
    total_loss = 0.0
    total_box = 0.0
    total_cls = 0.0
    total_dfl = 0.0

    num_samples = 0

    progress = tqdm(dataloader, desc="train", leave=False)  # tqdm으로 진행상황 시각화
    for images, targets in progress:  # 이미지, 타겟을 가져옴
        loss_batch = _targets_to_yolo_batch(
            targets, images, device
        )  # 타겟 데이터를 YOLO 모델에 맞게 변환
        images = torch.stack(images).to(
            device
        )  # 이미지를 배치 형태로 스택하고, device로 이동

        optimizer.zero_grad()
        output = model(images)

        num_samples += len(images)
        loss_vec, _ = criterion(output, loss_batch)
        loss = loss_vec.sum()
        loss.backward()
        optimizer.step()
        if ema is not None:     #ema가 켜져 있으면
            ema.update(model)       #업데이트

        loss_value = loss.item()
        total_loss += loss_value
        total_box += loss_vec[0].item()
        total_cls += loss_vec[1].item()
        total_dfl += loss_vec[2].item()

        progress.set_postfix(
            loss=f"{loss_value:.4f}",
            box=f"{loss_vec[0].item():.2f}",
            cls=f"{loss_vec[1].item():.2f}",
            dfl=f"{loss_vec[2].item():.2f}",
        )  # 진행 바 업데이트 시 loss 값을 표시

    return (
        total_loss / num_samples,
        total_box / num_samples,
        total_cls / num_samples,
        total_dfl / num_samples,
    )

