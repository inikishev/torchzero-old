"""Basic blocks"""
from typing import Optional, Literal
from collections.abc import Sequence, Callable
import torch
import math

from .generic_block import _create_module_order
from .._library.convolution import Convnd, ConvTransposend, LazyConvnd, LazyConvTransposend
from .._library.norm import create_norm
from .._library.dropout import create_dropout
from .._library.pool import create_pool
from .._library.activation import create_act
from .._library.upsample import create_upsample
from ..layers.pad import pad_to_shape, pad_like
from ..layers.crop import SpatialReduceCrop

__all__ = [
    'ConvBlock',
    'ConvTransposeBlock',
]

def _get_samesize_padding_int(kernel_size:int | Sequence[int]) -> int | tuple:
    """Return padding that retains input size given a kernel size."""
    if isinstance(kernel_size, int): return int((kernel_size - 1) // 2)
    else: return tuple([_get_samesize_padding_int(i) for i in kernel_size]) # type:ignore
    
def _get_samesize_padding_float(kernel_size:int | Sequence[int]) -> float | tuple:
    """Return padding that retains input size given a kernel size."""
    if isinstance(kernel_size, int): return (kernel_size - 1) / 2
    else: return tuple([_get_samesize_padding_float(i) for i in kernel_size]) # type:ignore


def _act_is_first(order:str, main_char:str):
    order = order.upper()
    for char in order:
        if char == 'A': return True
        elif char == main_char: return False

class ConvBlock(torch.nn.Module):
    def __init__(self,
        in_channels: Optional[int],
        out_channels: Optional[int],
        kernel_size: int | tuple ,
        stride: int | tuple  = 1,
        padding: float | tuple | Literal['auto'] = 'auto',
        dilation: int | tuple = 1,
        groups: int = 1,
        bias: bool | Literal['auto'] = 'auto',
        norm: Optional[torch.nn.Module | str | bool | Callable | str] = None,
        dropout: Optional[float | torch.nn.Module | Callable] = None,
        act: Optional[torch.nn.Module | Callable | str] = None,
        pool: Optional[int | torch.nn.Module | Callable | str] = None,
        upsample: Optional[int | torch.nn.Module | Callable | str] = None,
        residual = False,
        recurrent = 1,
        ndim: int = 2,
        order = "UCPAND",
        device = None,
        spatial_size: Optional[Sequence[int]] = None,
        conv: torch.nn.Module | Callable | Literal['torch'] = 'torch',
    ):
        """Convolution block.

        Default order is: convolution -> pooling -> activation -> normalization -> dropout.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            kernel_size (_type_): Kernel size, integer or tuple.
            stride (int, optional): Stride, integer or tuple. Defaults to 1.
            padding (int, optional): Padding, integer or tuple. Defaults to 0.
            dilation (int, optional): Dilation. Defaults to 1.
            groups (int, optional): Groups, input channels must be divisible by this (?). Defaults to 1.
            bias (bool, optional): Whether to enable convolution bias. Defaults to True.
            norm (Optional[torch.nn.Module  |  str  |  bool], optional): Norm, a module or just `True` for batch norm. Defaults to None.
            dropout (Optional[float  |  torch.nn.Module], optional): Dropout, a module or dropout probability float. Defaults to None.
            act (Optional[torch.nn.Module], optional): Activation function. Defaults to None.
            pool (Optional[int  |  torch.nn.Module], optional): Pooling, a module or an integer of max pooling kernel size and stride. Defaults to None.
            ndim (int, optional): Number of dimensions. Defaults to 2.
            custom_op (_type_, optional): Custom operation to replace convolution. Defaults to None.
            order (str, optional): Order of operations. Defaults to "cpand".
        """
        super().__init__()

        if out_channels is None:
            if in_channels is None: raise ValueError("Either `in_channels` or `out_channels` must be provided.")
            out_channels = in_channels

        if conv == 'torch':
            if in_channels is None: conv = LazyConvnd
            else: conv = Convnd

        if bias == 'auto':
            if norm is not None: bias = False
            else: bias = True

        if padding == 'auto': padding = _get_samesize_padding_float(kernel_size)
        if isinstance(padding, float):
            if padding % 1 == 0: padding = int(padding); crop = None
            elif padding % 1 == 0.5: padding = int(math.ceil(padding)); crop = SpatialReduceCrop(1)
            else: raise ValueError(f'Invalid padding: {padding}')
        else: crop = None

        # convolution
        conv_layer = conv(
                in_channels=in_channels, # type:ignore
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias, # type:ignore
                device=device,
                ndim = ndim,
            )

        self.layers = _create_module_order(
            modules = dict(C=conv_layer, P=pool, A=act, N=norm, D=dropout, U=upsample, _=crop),
            order = order.replace("C", "C_"),
            main_module='C',
            in_channels = in_channels,
            out_channels=out_channels,
            ndim = ndim,
            spatial_size = spatial_size,
            )

        self.residual = residual
        self.recurrent = recurrent

    def forward(self, x:torch.Tensor):
        for _ in range(self.recurrent):
            if self.residual: x = x + pad_like(self.layers(x), x)
            else: x = self.layers(x)
        return x

def _get_samesize_transpose_padding(kernel_size:int | Sequence[int]) -> int | tuple:
    """Return padding that retains input size given a kernel size."""
    if isinstance(kernel_size, int): return int((kernel_size - 1) // 2)
    else: return tuple([_get_samesize_transpose_padding(i) for i in kernel_size]) # type:ignore

class ConvTransposeBlock(torch.nn.Module):
    def __init__(self,
        in_channels: Optional[int],
        out_channels: Optional[int],
        kernel_size: int | tuple,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
        output_padding: int | tuple = 0,
        groups = 1,
        bias: bool | Literal['auto'] = 'auto',
        dilation: int = 1,
        norm: Optional[torch.nn.Module | str | bool | Callable] = None,
        dropout: Optional[float | torch.nn.Module | Callable] = None,
        act: Optional[torch.nn.Module | Callable] = None,
        upsample: Optional[float | torch.nn.Module | Callable] = None,
        pool: Optional[int | torch.nn.Module | Callable] = None,
        ndim: int = 2,
        order = "UCPAND",
        device=None,
        spatial_size: Optional[Sequence[int]] = None,
        upconv: torch.nn.Module | Callable | Literal['torch'] = 'torch'
    ):
        """Transposed convolution block.

        Default order is: upsample -> transposed convolution -> activation -> normalization -> dropout.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            kernel_size (_type_): Kernel size, integer or tuple.
            stride (int, optional): Stride, integer or tuple. Defaults to 1.
            padding (int, optional): Padding, integer or tuple. Defaults to 0.
            output_padding (int, optional): Output padding. Defaults to 0.
            groups (int, optional): Groups, input channels must be divisible by this (?). Defaults to 1.
            bias (bool, optional): Whether to enable convolution bias. Defaults to True.
            dilation (int, optional): Dilation. Defaults to 1.
            norm (Optional[torch.nn.Module  |  str  |  bool], optional): Norm, a module or just `True` for batch norm. Defaults to None.
            dropout (Optional[float  |  torch.nn.Module], optional): Dropout, a module or dropout probability float. Defaults to None.
            act (Optional[torch.nn.Module], optional): Activation function. Defaults to None.
            upsample (Optional[int  |  torch.nn.Module], optional): Upscaling, a module or float of upscaling factor. Defaults to None.
            ndim (int, optional): Number of dimensions. Defaults to 2.
            custom_op (_type_, optional): Custom operation to replace convolution. Defaults to None.
            order (str, optional): Order of operations. Defaults to "cpand".
        """
        super().__init__()

        if out_channels is None:
            if in_channels is None: raise ValueError("Either `in_channels` or `out_channels` must be provided.")
            out_channels = in_channels

        if upconv == 'torch':
            if in_channels is None: upconv = LazyConvTransposend
            else: upconv = ConvTransposend

        if bias == 'auto':
            if norm is not None: bias = False
            else: bias = True

        # convolution
        upconv_layer = upconv(
            in_channels=in_channels, # type:ignore
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias, # type:ignore
            dilation=dilation,
            device=device,
            ndim = ndim,
        )

        self.layers = _create_module_order(
            modules = dict(C=upconv_layer, U=upsample, A=act, N=norm, D=dropout, P=pool),
            order = order,
            main_module='C',
            in_channels = in_channels,
            out_channels=out_channels,
            ndim = ndim,
            spatial_size = spatial_size,
            )

    def forward(self, x:torch.Tensor):
        return self.layers(x)
