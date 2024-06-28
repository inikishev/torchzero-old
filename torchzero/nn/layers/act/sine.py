from typing import Optional
from torch import nn
import torch.nn.functional as F
import torch

__all__ = [
    'SineAct',
]

class SineAct(nn.Module):
    """`sin` act function wrapped into an `nn.Module` for convenience, calls `torch.sin(x)`."""
    def __init__(self):
        super().__init__()
    def forward(self, x): return torch.sin(x)