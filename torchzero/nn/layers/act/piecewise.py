from typing import Optional
from torch import nn
import torch.nn.functional as F
import torch

__all__ = [
    'VShaped',
    "v_shaped",
]

FloatOrTensor = int | float | torch.Tensor


def v_shaped(x, threshold: FloatOrTensor = 0., inverse = False):
    """A simple piecewise V-shaped activation function"""
    if isinstance(threshold, (int, float)): threshold = torch.tensor(threshold)
    if inverse: return torch.maximum(x, threshold) - torch.minimum(x, threshold)
    return torch.minimum(x, threshold) - torch.maximum(x, threshold)

class VShaped(nn.Module):
    """A simple piecewise V-shaped activation function"""
    def __init__(self, threshold = 0., inverse = False):
        super().__init__()
        self.threshold = torch.tensor(threshold)
        self.inverse = inverse
    def forward(self, x): return v_shaped(x, self.threshold, self.inverse)


