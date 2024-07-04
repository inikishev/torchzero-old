"""ok"""
from typing import Any, Optional
from collections.abc import Callable
import torch
import torch.nn.common_types
from ..._library.convolution import Convnd, ConvTransposend
def identity(x): return x
def _identity_if_none(x):
    if x is None: return identity
    else: return x

__all__ = [
    "RMSConv",
    "ArctanConv",
    "RMSConvTranspose",
    "ArctanConvTranspose",
]
class RMSConv(torch.nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device: Any | None = None,
        dtype: Any | None = None,
        n = 2,
        ndim=2,
        act:Optional[Callable] = None,
        conv = Convnd,
):
        super().__init__()
        self.conv_modules = torch.nn.ModuleList([conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            ndim = ndim
        ) for _ in range(n)])
        self.act = _identity_if_none(act)

    def forward(self, x):
        for i, m in enumerate(self.conv_modules):
            if i == 0: res = self.act(m(x))**2
            else: res += self.act(m(x))**2 # type:ignore
        return res ** 0.5 # type:ignore


class ArctanConv(torch.nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device: Any | None = None,
        dtype: Any | None = None,
        ndim = 2,
        act:Optional[Callable] = None,
        conv = Convnd,
):
        super().__init__()
        self.conv_modules = torch.nn.ModuleList([conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            ndim = ndim,
        ) for _ in range(2)])
        self.act = _identity_if_none(act)

    def forward(self, x):
        return torch.arctan2(self.act(self.conv_modules[0](x)), self.act(self.conv_modules[1](x)))


class RMSConvTranspose(torch.nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        output_padding = 0,
        groups: int = 1,
        bias: bool = True,
        dilation = 1,
        padding_mode: str = 'zeros',
        device = None,
        dtype = None,
        n = 2,
        ndim=2,
        act:Optional[Callable] = None,
        conv = ConvTransposend,
):
        super().__init__()
        self.conv_modules = torch.nn.ModuleList([conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            ndim = ndim,
        ) for _ in range(n)])
        self.act = _identity_if_none(act)

    def forward(self, x):
        for i, m in enumerate(self.conv_modules):
            if i == 0: res = self.act(m(x))**2
            else: res += self.act(m(x))**2 # type:ignore
        return res ** 0.5 # type:ignore


class ArctanConvTranspose(torch.nn.Module):
    def __init__(self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride = 1,
            padding = 0,
            output_padding = 0,
            groups: int = 1,
            bias: bool = True,
            dilation = 1,
            padding_mode: str = 'zeros',
            device = None,
            dtype = None,
            ndim=2,
            act:Optional[Callable] = None,
            conv = ConvTransposend,
        ):
        super().__init__()
        self.conv_modules = torch.nn.ModuleList([conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            ndim = ndim,
        ) for _ in range(2)])
        self.act = _identity_if_none(act)

    def forward(self, x):
        return torch.arctan2(self.act(self.conv_modules[0](x)), self.act(self.conv_modules[1](x)))