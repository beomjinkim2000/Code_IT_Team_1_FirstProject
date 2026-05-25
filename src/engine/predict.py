"""Prediction skeleton
이 모듈은 1 batch 단위로 model forward를 수행 raw output을 반환.
confidence filtering, NMS, max_detections 제한, pred_dict 생성은
src/engine/postprocess.py에서 담당
"""
from __future__ import annotations

from typing import Any

import torch

def predict_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device | str,
) -> Any:
   """한 배치에 모델 추론 실행 원본 모델 출력을 반환하는 함수
    
    todo(issue #18): 실제 YOLO 모델 output 형식이 확정되면
    postprocess.py가 받을 수 있는 raw output 형태를 문서화
    
    model.eval()은 루트 predict.py에서 수행
    """
   
   with torch.no_grad(): #추론에서는 그라 계산x 메모리 절약 위해 no_grad
     images = images.to(device)

     raw_outputs = model(images) #실제 모델 추론 결과
   return raw_outputs
