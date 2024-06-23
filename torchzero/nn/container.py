from collections.abc import Callable
from torch import nn
import torch

__all__ = [
    'FuncModule',
    'ensure_module',
]

class FuncModule(nn.Module):
    def __init__(self, func:Callable):
        """Wrap `func` into a torch.nn.Module.

        Args:
            func (Callable): _description_
        """
        super().__init__()
        self.func = func
    def forward(self, x): return self.func(x)

def ensure_module(x) -> nn.Module:
    if isinstance(x, nn.Module): return x
    elif isinstance(x, Callable): return FuncModule(x)
    else: raise TypeError("Can't convert to module")