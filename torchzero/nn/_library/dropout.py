from collections.abc import Sequence
from typing import Optional, Any
import torch

def get_dropoutnd(ndim:int):
    if ndim == 0: return torch.nn.Dropout
    if ndim == 1: return torch.nn.Dropout1d
    elif ndim == 2: return torch.nn.Dropout2d
    elif ndim == 3: return torch.nn.Dropout3d
    else: raise NotImplementedError(f'get_dropoutnd only supports 1-3d dropout, got {ndim}')

def create_dropout(module:Any, num_channels:Optional[int], ndim:Optional[int], spatial_size:Optional[Sequence[int]], p:Optional[int] = None):
    if callable(module) or module is None: return module

    if isinstance(module, str): module = module.lower().strip()

    if isinstance(module, float):
        if ndim is None: raise ValueError('ndim must be specified when using float dropout rate')
        return get_dropoutnd(ndim)(module)

    else: raise ValueError(f'Unknown dropout module: {module}')