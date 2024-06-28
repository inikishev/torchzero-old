from collections.abc import Sequence
from typing import Optional, Any
import torch

def get_batchnormnd(ndim):
    if ndim == 1: return torch.nn.BatchNorm1d
    elif ndim == 2: return torch.nn.BatchNorm2d
    elif ndim == 3: return torch.nn.BatchNorm3d
    else: raise NotImplementedError(f'get_batchnormnd only supports 1-3d batch-norm, got {ndim}')

def get_lazybatchnormnd(ndim):
    if ndim == 1: return torch.nn.LazyBatchNorm1d
    elif ndim == 2: return torch.nn.LazyBatchNorm2d
    elif ndim == 3: return torch.nn.LazyBatchNorm3d
    else: raise NotImplementedError(f'get_batchnormnd only supports 1-3d batch-norm, got {ndim}')

def get_instancenormnd(ndim):
    if ndim == 1: return torch.nn.InstanceNorm1d
    elif ndim == 2: return torch.nn.InstanceNorm2d
    elif ndim == 3: return torch.nn.InstanceNorm3d
    else: raise NotImplementedError(f'get_batchnormnd only supports 1-3d batch-norm, got {ndim}')


def get_lazyinstancenormnd(ndim):
    if ndim == 1: return torch.nn.LazyInstanceNorm1d
    elif ndim == 2: return torch.nn.LazyInstanceNorm2d
    elif ndim == 3: return torch.nn.LazyInstanceNorm3d
    else: raise NotImplementedError(f'get_batchnormnd only supports 1-3d batch-norm, got {ndim}')


def create_norm(module:Any, num_channels:Optional[int], ndim:Optional[int], spatial_size:Optional[Sequence[int]]):
    if callable(module) or module is None: return module

    if isinstance(module, str): module = module.lower().strip()

    if module in ('b','bn', 'batch', 'batchnorm','batch norm', True): 
        if num_channels is not None: return get_batchnormnd(ndim)(num_features=num_channels)
        else: return get_lazybatchnormnd(ndim)()
    if module in ('i', 'in', 'instance', 'instancenorm', 'instance norm'): 
        if num_channels is not None: return get_instancenormnd(ndim)(num_features=num_channels)
        else: return get_lazyinstancenormnd(ndim)()
    if module in ('l', 'ln', 'layer', 'layernorm', 'layer norm'): 
        if spatial_size is None: raise ValueError('spatial_size must be specified when using layernorm')
        if num_channels is None: raise ValueError('num_channels must be specified when using layernorm')
        return torch.nn.LayerNorm(normalized_shape = list((num_channels, *spatial_size)))
    raise ValueError(f'Unknown norm module: {module}')