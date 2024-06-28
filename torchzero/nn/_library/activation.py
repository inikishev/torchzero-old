from collections.abc import Sequence
from typing import Optional, Any
import torch

def identity(x): return x

def create_act(module:Any, num_channels:Optional[int], ndim:Optional[int], spatial_size:Optional[Sequence[int]]):
    if module is None: return None
    if callable(module): return module

    if isinstance(module, str): x = module.lower().strip()

    if module in ('relu',): return torch.nn.ReLU(inplace=True)
    if module in ('leakyrelu',): return torch.nn.LeakyReLU(0.1, inplace=True)
    if module in ('identity',): return identity
    else: raise ValueError(f'Unknown activation module: {module}')
