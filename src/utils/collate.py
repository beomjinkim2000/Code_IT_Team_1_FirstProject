from typing import List, Tuple, Dict
import torch
from torch import Tensor


def collate_fn(batch) -> Tuple[List[Tensor], List[Dict]]:
    images, targets = zip(*batch)
    return list(images), list(targets)
