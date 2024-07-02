from collections.abc import Sequence
from typing import Optional, Any
import torch
from torch import nn

def get_maxpoolnd(ndim:int):
    if ndim == 1: return torch.nn.MaxPool1d
    elif ndim == 2: return torch.nn.MaxPool2d
    elif ndim == 3: return torch.nn.MaxPool3d
    else: raise NotImplementedError(f'get_maxpoolnd only supports 1-3d max-pooling, got {ndim}')

def get_avgpoolnd(ndim:int):
    if ndim == 1: return torch.nn.AvgPool1d
    elif ndim == 2: return torch.nn.AvgPool2d
    elif ndim == 3: return torch.nn.AvgPool3d
    else: raise NotImplementedError(f'get_maxpoolnd only supports 1-3d max-pooling, got {ndim}')

def get_fractionalmaxpoolnd(ndim:int):
    # no 1d
    if ndim == 2: return torch.nn.FractionalMaxPool2d
    elif ndim == 3: return torch.nn.FractionalMaxPool3d
    else: raise NotImplementedError(f'get_maxpoolnd only supports 1-3d max-pooling, got {ndim}')
    
def get_adaptiveavgpoolnd(ndim:int):
    if ndim == 1: return torch.nn.AdaptiveAvgPool1d
    elif ndim == 2: return torch.nn.AdaptiveAvgPool2d
    elif ndim == 3: return torch.nn.AdaptiveAvgPool3d
    else: raise NotImplementedError(f'get_maxpoolnd only supports 1-3d max-pooling, got {ndim}')

def get_adaptivemaxpoolnd(ndim:int):
    if ndim == 1: return torch.nn.AdaptiveMaxPool1d
    elif ndim == 2: return torch.nn.AdaptiveMaxPool2d
    elif ndim == 3: return torch.nn.AdaptiveMaxPool3d
    else: raise NotImplementedError(f'get_maxpoolnd only supports 1-3d max-pooling, got {ndim}')

def create_pool(module:Any, num_channels:Optional[int], ndim:Optional[int], spatial_size:Optional[Sequence[int]], scale:Any = 0.5):
    reduction = 1 / scale
    if callable(module) or module is None: return module
    if module in ('max', 'pool', 'maxpool'): 
        if ndim is None: raise ValueError('ndim must be specified when using maxpool')
        return get_maxpoolnd(ndim)(int(reduction), int(reduction))
    elif module in ('avg', 'avgpool'): 
        if ndim is None: raise ValueError('ndim must be specified when using avgpool')
        return get_avgpoolnd(ndim)(int(reduction), int(reduction))
    elif module in ('frac', 'fracpool', 'fractional'): 
        if ndim is None: raise ValueError('ndim must be specified when using fractional maxpool')
        return get_fractionalmaxpoolnd(ndim)(reduction)
    else: raise ValueError(f'Unknown pooling module: {module}')

def MaxPoolnd(
    kernel_size: int | tuple,
    stride: int | tuple | None = None,
    padding: int | tuple = 0,
    dilation: int | tuple = 1,
    return_indices: bool = False,
    ceil_mode: bool = False,
    ndim: int = 2,
    ):
    kwargs = locals().copy()
    kwargs.pop('ndim')
    return get_maxpoolnd(ndim)(**kwargs)

def AvgPoolnd(
    kernel_size: int | tuple,
    stride: int | tuple | None = None,
    padding: int | tuple = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
    ndim: int = 2,
    ):
    kwargs = locals().copy()
    kwargs.pop('ndim')
    return get_avgpoolnd(ndim)(**kwargs)

def FractionalMaxPoolnd(kernel_size: int | tuple,
    output_size: int | tuple | None = None,
    output_ratio = None,
    return_indices: bool = False,
    _random_samples = None,
    ndim:int = 2
    ):
    kwargs = locals().copy()
    kwargs.pop('ndim')
    return get_fractionalmaxpoolnd(ndim)(**kwargs)

def AdaptiveAvgPoolnd(
    output_size: int | tuple,
    ndim:int = 2
    ):
    return get_adaptiveavgpoolnd(ndim)(output_size=output_size)

def AdaptiveMaxPoolnd(
    output_size: int | tuple,
    return_indices: bool = False,
    ndim:int = 2
    ):
    return get_adaptivemaxpoolnd(ndim)(output_size=output_size, return_indices=return_indices)

