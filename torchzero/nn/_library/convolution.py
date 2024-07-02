import torch
from torch import nn


def get_convnd(ndim:int):
    if ndim == 1: return torch.nn.Conv1d
    elif ndim == 2: return torch.nn.Conv2d
    elif ndim == 3: return torch.nn.Conv3d
    else: raise NotImplementedError(f'get_convnd only supports 1-3d convolutions, got {ndim}')

def get_convtransposednd(ndim:int):
    if ndim == 1: return torch.nn.ConvTranspose1d
    elif ndim == 2: return torch.nn.ConvTranspose2d
    elif ndim == 3: return torch.nn.ConvTranspose3d
    else: raise NotImplementedError(f'get_convtransposednd only supports 1-3d transposed convolutions, got {ndim}')

def get_lazyconvnd(ndim:int):
    if ndim == 1: return torch.nn.LazyConv1d
    elif ndim == 2: return torch.nn.LazyConv2d
    elif ndim == 3: return torch.nn.LazyConv3d
    else: raise NotImplementedError(f'get_lazyconvnd only supports 1-3d convolutions, got {ndim}')

def get_lazyconvtransposend(ndim:int):
    if ndim == 1: return torch.nn.LazyConvTranspose1d
    elif ndim == 2: return torch.nn.LazyConvTranspose2d
    elif ndim == 3: return torch.nn.LazyConvTranspose3d
    else: raise NotImplementedError(f'get_lazyconvtransposend only supports 1-3d transposed convolutions, got {ndim}')

def Convnd(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple,
    stride: int | tuple = 1,
    padding: int | tuple | str = 0,
    dilation: int | tuple = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: str = "zeros",
    device=None,
    dtype=None,
    ndim=2,
):
    """Same as using `torch.nn.Conv*d`. Creates either a `Conv1d`, `Conv2d` or `Conv3d` depending on `ndim`.

    Args:
        in_channels (int): _description_
        out_channels (int): _description_
        kernel_size (int | tuple): _description_
        stride (int | tuple, optional): _description_. Defaults to 1.
        padding (int | tuple | str, optional): _description_. Defaults to 0.
        dilation (int | tuple, optional): _description_. Defaults to 1.
        groups (int, optional): _description_. Defaults to 1.
        bias (bool, optional): _description_. Defaults to True.
        padding_mode (str, optional): _description_. Defaults to "zeros".
        device (_type_, optional): _description_. Defaults to None.
        dtype (_type_, optional): _description_. Defaults to None.
        ndim (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    kwargs = locals().copy()
    kwargs.pop('ndim')
    return get_convnd(ndim)(**kwargs)

def ConvTransposend(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple,
    stride: int | tuple = 1,
    padding: int | tuple = 0,
    output_padding: int | tuple = 0,
    groups: int = 1,
    bias: bool = True,
    dilation: int | tuple = 1,
    padding_mode: str = "zeros",
    device=None,
    dtype=None,
    ndim=2,
):
    kwargs = locals().copy()
    kwargs.pop('ndim')
    return get_convtransposednd(ndim)(**kwargs)


def LazyConvnd(
    out_channels: int,
    kernel_size: int | tuple,
    stride: int | tuple = 1,
    padding: int | tuple = 0,
    dilation: int | tuple = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: str = "zeros",
    device=None,
    dtype=None,
    ndim=2,
    in_channels = None,
):
    kwargs = locals().copy()
    kwargs.pop('in_channels')
    kwargs.pop('ndim')
    return get_lazyconvnd(ndim)(**kwargs)


def LazyConvTransposend(
    out_channels: int,
    kernel_size: int | tuple,
    stride: int | tuple = 1,
    padding: int | tuple = 0,
    output_padding: int | tuple = 0,
    groups: int = 1,
    bias: bool = True,
    dilation: int = 1,
    padding_mode: str = "zeros",
    device=None,
    dtype=None,
    ndim=2,
    in_channels = None,
):
    kwargs = locals().copy()
    kwargs.pop('in_channels')
    kwargs.pop('ndim')
    return get_lazyconvtransposend(ndim)(**kwargs)
