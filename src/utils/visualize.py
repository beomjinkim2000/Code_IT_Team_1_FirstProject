from typing import List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import Tensor


def draw_boxes(
    image: Image.Image,
    boxes: Tensor,
    labels: Optional[List[str]] = None,
    scores: Optional[Tensor] = None,
    color: str = "red",
) -> Image.Image:
    """이미지에 바운딩 박스 오버레이. boxes: [N, 4] xyxy 절대 픽셀"""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        parts = []
        if labels is not None:
            parts.append(labels[i])
        if scores is not None:
            parts.append(f"{scores[i].item():.2f}")
        if parts:
            draw.text((x1, max(y1 - 12, 0)), " ".join(parts), fill=color)

    return img
