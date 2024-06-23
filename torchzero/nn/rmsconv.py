"""Experimental root mean square convolutional blocks (work worse then normal convolutions)"""
from typing import Any, Optional
from collections.abc import Callable
import torch
import torch.nn.common_types

def identity(x): return x


__all__ = [
    'RMSConv',
    'ArctanConv',
    'RMSConvTranspose',
    'ArctanConvTranspose',
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
        act = None,
        bias: bool = True,
        padding_mode: str = 'zeros',
        n = 2,
        ndim=2,
        device: Any | None = None,
        dtype: Any | None = None,
):
        """Uses `n` convolutional layers, takes a root mean square of their outputs.

        If `act` is specified, that activation function is applied to convolutional layer outputs before taking root mean square.

        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            kernel_size (_type_): _description_
            stride (int, optional): _description_. Defaults to 1.
            padding (int, optional): _description_. Defaults to 0.
            dilation (int, optional): _description_. Defaults to 1.
            groups (int, optional): _description_. Defaults to 1.
            act (Optional[Callable], optional): _description_. Defaults to None.
            bias (bool, optional): _description_. Defaults to True.
            padding_mode (str, optional): _description_. Defaults to 'zeros'.
            n (int, optional): _description_. Defaults to 2.
            ndim (int, optional): _description_. Defaults to 2.
            device (Any | None, optional): _description_. Defaults to None.
            dtype (Any | None, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        super().__init__()
        if ndim == 1: Convnd = torch.nn.Conv1d
        elif ndim == 2: Convnd = torch.nn.Conv2d
        elif ndim == 3: Convnd = torch.nn.Conv3d
        else: raise ValueError
        self.conv_modules = torch.nn.ModuleList([Convnd(
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
        ) for _ in range(n)])

        self.act = act if act is not None else identity

    def forward(self, x):
        for i, m in enumerate(self.conv_modules):
            if i == 0: res = self.act(m(x)) ** 2
            else: res += self.act(m(x)) ** 2 # type:ignore
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
        act:Optional[Callable] = None,
        padding_mode: str = 'zeros',
        ndim = 2,
        device: Any | None = None,
        dtype: Any | None = None,
        ):
        """Uses 2 convolutional layers, takes the arctan2 of their outputs.

        If `act` is specified, that activation function is applied to convolutional layer outputs before taking arctan2.

        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            kernel_size (_type_): _description_
            stride (int, optional): _description_. Defaults to 1.
            padding (int, optional): _description_. Defaults to 0.
            dilation (int, optional): _description_. Defaults to 1.
            groups (int, optional): _description_. Defaults to 1.
            bias (bool, optional): _description_. Defaults to True.
            act (Optional[Callable], optional): _description_. Defaults to None.
            padding_mode (str, optional): _description_. Defaults to 'zeros'.
            ndim (int, optional): _description_. Defaults to 2.
            device (Any | None, optional): _description_. Defaults to None.
            dtype (Any | None, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        super().__init__()
        if ndim == 1: Convnd = torch.nn.Conv1d
        elif ndim == 2: Convnd = torch.nn.Conv2d
        elif ndim == 3: Convnd = torch.nn.Conv3d
        else: raise ValueError
        self.conv_modules = torch.nn.ModuleList([Convnd(
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
        ) for _ in range(2)])
        self.act = act if act is not None else identity

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
        act:Optional[Callable] = None,
        n = 2,
        ndim=2,
        device = None,
        dtype = None,
):
        """Uses `n` convolutional layers, takes a root mean square of their outputs.

        If `act` is specified, that activation function is applied to convolutional layer outputs before taking root mean square.

        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            kernel_size (_type_): _description_
            stride (int, optional): _description_. Defaults to 1.
            padding (int, optional): _description_. Defaults to 0.
            output_padding (int, optional): _description_. Defaults to 0.
            groups (int, optional): _description_. Defaults to 1.
            bias (bool, optional): _description_. Defaults to True.
            dilation (int, optional): _description_. Defaults to 1.
            padding_mode (str, optional): _description_. Defaults to 'zeros'.
            act (Optional[Callable], optional): _description_. Defaults to None.
            n (int, optional): _description_. Defaults to 2.
            ndim (int, optional): _description_. Defaults to 2.
            device (_type_, optional): _description_. Defaults to None.
            dtype (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        super().__init__()
        if ndim == 1: Convnd = torch.nn.ConvTranspose1d
        elif ndim == 2: Convnd = torch.nn.ConvTranspose2d
        elif ndim == 3: Convnd = torch.nn.ConvTranspose3d
        else: raise ValueError
        self.conv_modules = torch.nn.ModuleList([Convnd(
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
        ) for _ in range(n)])
        self.act = act if act is not None else identity

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
        ):
        """Uses 2 convolutional layers, takes the arctan2 of their outputs.

        If `act` is specified, that activation function is applied to convolutional layer outputs before taking arctan2.

        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            kernel_size (_type_): _description_
            stride (int, optional): _description_. Defaults to 1.
            padding (int, optional): _description_. Defaults to 0.
            output_padding (int, optional): _description_. Defaults to 0.
            groups (int, optional): _description_. Defaults to 1.
            bias (bool, optional): _description_. Defaults to True.
            dilation (int, optional): _description_. Defaults to 1.
            padding_mode (str, optional): _description_. Defaults to 'zeros'.
            device (_type_, optional): _description_. Defaults to None.
            dtype (_type_, optional): _description_. Defaults to None.
            ndim (int, optional): _description_. Defaults to 2.
            act (Optional[Callable], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        super().__init__()
        if ndim == 1: Convnd = torch.nn.ConvTranspose1d
        elif ndim == 2: Convnd = torch.nn.ConvTranspose2d
        elif ndim == 3: Convnd = torch.nn.ConvTranspose3d
        else: raise ValueError
        self.conv_modules = torch.nn.ModuleList([Convnd(
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
        ) for _ in range(2)])
        self.act = act if act is not None else identity

    def forward(self, x):
        return torch.arctan2(self.act(self.conv_modules[0](x)), self.act(self.conv_modules[1](x)))