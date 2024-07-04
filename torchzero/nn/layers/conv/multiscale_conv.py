"""ok"""
from typing import Any, Optional
from collections.abc import Callable
import torch
import torch.nn.common_types
from ..._library.convolution import Convnd
from ..act.multiscale import Multiscale

def identity(x): return x
def _identity_if_none(x):
    if x is None: return identity
    else: return x

__all__ = [
    "MultiscaleConv",
]

class MultiscaleConv(torch.nn.Sequential):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        device: Any | None = None,
        dtype: Any | None = None,
        ndim=2,
        conv = Convnd,
        learnable_scales = True,
):
        """Multiscale convolution, must be followed by some nonlinearity.

        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            kernel_size (_type_): _description_
            stride (int, optional): _description_. Defaults to 1.
            padding (int, optional): _description_. Defaults to 0.
            dilation (int, optional): _description_. Defaults to 1.
            groups (int, optional): _description_. Defaults to 1.
            bias (bool, optional): _description_. Defaults to True.
            device (Any | None, optional): _description_. Defaults to None.
            dtype (Any | None, optional): _description_. Defaults to None.
            ndim (int, optional): _description_. Defaults to 2.
            conv (_type_, optional): _description_. Defaults to Convnd.
            learnable_scales (bool, optional): _description_. Defaults to True.

        Raises:
            ValueError: _description_
        """
        conv = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
            device=device,
            dtype=dtype,
            ndim = ndim,
        )
        expand_scale = out_channels / in_channels
        if expand_scale % 1 != 0:
            raise ValueError(f"out_channels must be divisible by in_channels, {out_channels = }, {in_channels = }")
        weights = torch.linspace(start=-1.5, end=1.5, steps=int(expand_scale), device=device, dtype=dtype)
        multiscale = Multiscale(weights, torch.zeros_like(weights), learnable=learnable_scales)

        super().__init__(conv, multiscale)