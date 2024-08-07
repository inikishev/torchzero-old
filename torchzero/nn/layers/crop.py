from collections.abc import Sequence
from typing import Literal
import torch

from ..functional.crop import spatial_crop
__all__ = [
    "SpatialCrop"
]

class SpatialCrop(torch.nn.Module):
    def __init__(self, crop:int = 1):
        super().__init__()
        self.crop = crop

    def forward(self, x:torch.Tensor):
        return spatial_crop(x, crop=self.crop)