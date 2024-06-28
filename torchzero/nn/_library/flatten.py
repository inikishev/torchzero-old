from collections.abc import Sequence
from typing import Optional, Any
import torch
from torch import nn

def create_flatten(module, num_channels:Optional[int], ndim:Optional[int], spatial_size:Optional[Sequence[int]], dims:Optional[int | Sequence[int]] = None):
    if callable(module) or module is None: return module
    if module is True: 
        if dims is None: start_dim, end_dim = 1, -1
        elif isinstance(dims, int): start_dim, end_dim = dims, dims
        else: start_dim, end_dim = dims
        return torch.nn.Flatten(start_dim=start_dim, end_dim=end_dim)
    else: raise ValueError(f'Unknown upsample module: {module}')

