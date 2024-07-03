"""Basic blocks"""
from typing import Optional, Literal
from collections.abc import Sequence, Callable
import functools
import torch
from ..layers.conv import ConvTransposeBlock
from .._library.pool import create_pool
from ._utils import _get_partial_from_locals


__all__ = [
    "ConvTranspose2StrideCube",
    "ConvTranspose3StrideCube",
]

class ConvTranspose2StrideCube(ConvTransposeBlock):
    """Convolution block"""
    def __init__(self,
        in_channels: Optional[int],
        out_channels: Optional[int] = None,
        scale: Optional[int] = 2,
        ndim: int = 2,
        act: Optional[torch.nn.Module | Callable | str] = None,
        norm: Optional[torch.nn.Module | str | bool | Callable] = None,
        dropout: Optional[float | torch.nn.Module | Callable] = None,
        bias: bool | Literal['auto'] = 'auto',
        groups = 1,
        order = "UCPAND",
        device=None,
        spatial_size: Optional[Sequence[int]] = None,
        upconv: torch.nn.Module | Callable | Literal['torch'] = 'torch'
        # resample: Optional[Literal["stride", "pool", "upsample"] | str | Callable | torch.nn.Module] = None,
        # padding: int | tuple = 0,
        # output_padding: int | tuple = 0,
        # dilation: int = 1,
        # upsample: Optional[float | torch.nn.Module | Callable] = None,
        # pool: Optional[int | torch.nn.Module | Callable] = None,
        ):
        """This is essentially a wrapper around ConvTransposeBlock that uses Cube interface, use ConvTransposeBlock directly if you need more sensible params.

        Args:
            in_channels (Optional[int]): Input channels, if `None` uses LazyConvnd
            out_channels (Optional[int], optional): Output channels, if `None` uses `input_channels`. Defaults to None.
            scale (Optional[float], optional): Input spatial size will be changed by this value (`kernel_size` and `stride` will be set to this). Defaults to None.
            groups (int, optional): _description_. Defaults to 1.
            bias (bool | Literal[&#39;auto&#39;], optional): On `auto`, disables bias if `norm` is not `None`. 
            Make sure to set this to `True` if you are using non-learnable norm. Defaults to 'auto'.
            norm (Optional[torch.nn.Module  |  str  |  bool  |  Callable], optional): _description_. Defaults to None.
            dropout (Optional[float  |  torch.nn.Module  |  Callable], optional): Dropout probability float, or a module. Defaults to None.
            act (Optional[torch.nn.Module  |  Callable], optional): _description_. Defaults to None.
            ndim (int, optional): Number of dimensions in input. Defaults to 2.
            order (str, optional): Order of operations by their first letters, 
            by default upscale -> convolution -> pooling -> activation -> norm -> dropout. Defaults to "UCPAND".
            device (_type_, optional): Something like `torch.device("cuda")`. Defaults to None.
            spatial_size (Optional[Sequence[int]], optional): Provide input spatial size, 
            sometimes needed if you use `LayerNorm` for example. Defaults to None.
            conv (torch.nn.Module | Callable | Literal[&#39;torch&#39;], optional): Module for convolution. 
            Defaults to 'torch' for ConvTransposend or LazyConvTransposend.
        """

        kwargs = locals().copy()
        kwargs.pop('self')
        kwargs.pop('__class__')

        kwargs['kernel_size'] = scale
        kwargs['stride'] = scale
        kwargs.pop('scale')

        super().__init__(**kwargs)

    @classmethod
    def partial(cls, # type:ignore
        act: Optional[torch.nn.Module | Callable | str] = None,
        norm: Optional[torch.nn.Module | str | bool | Callable] = None,
        dropout: Optional[float | torch.nn.Module | Callable] = None,
        bias: bool | Literal['auto'] = 'auto',
        groups = 1,
        order = "UCPAND",
        device=None,
        spatial_size: Optional[Sequence[int]] = None,
        upconv: torch.nn.Module | Callable | Literal['torch'] = 'torch',
        ):

        kwargs = locals().copy()
        kwargs.pop('cls')

        return functools.partial(cls, **kwargs)

class ConvTranspose3StrideCube(ConvTransposeBlock):
    """Convolution block"""
    def __init__(self,
        in_channels: Optional[int],
        out_channels: Optional[int] = None,
        scale: Optional[int] = 2,
        ndim: int = 2,
        act: Optional[torch.nn.Module | Callable | str] = None,
        norm: Optional[torch.nn.Module | str | bool | Callable] = None,
        dropout: Optional[float | torch.nn.Module | Callable] = None,
        bias: bool | Literal['auto'] = 'auto',
        groups = 1,
        order = "UCPAND",
        device=None,
        spatial_size: Optional[Sequence[int]] = None,
        upconv: torch.nn.Module | Callable | Literal['torch'] = 'torch'
        # resample: Optional[Literal["stride", "pool", "upsample"] | str | Callable | torch.nn.Module] = None,
        # padding: int | tuple = 0,
        # output_padding: int | tuple = 0,
        # dilation: int = 1,
        # upsample: Optional[float | torch.nn.Module | Callable] = None,
        # pool: Optional[int | torch.nn.Module | Callable] = None,
        ):
        """This is essentially a wrapper around ConvTransposeBlock that uses Cube interface, use ConvTransposeBlock directly if you need more sensible params.

        Args:
            in_channels (Optional[int]): Input channels, if `None` uses LazyConvnd
            out_channels (Optional[int], optional): Output channels, if `None` uses `input_channels`. Defaults to None.
            scale (Optional[float], optional): Input spatial size will be changed by this value (`kernel_size` and `stride` will be set to this). Defaults to None.
            groups (int, optional): _description_. Defaults to 1.
            bias (bool | Literal[&#39;auto&#39;], optional): On `auto`, disables bias if `norm` is not `None`. 
            Make sure to set this to `True` if you are using non-learnable norm. Defaults to 'auto'.
            norm (Optional[torch.nn.Module  |  str  |  bool  |  Callable], optional): _description_. Defaults to None.
            dropout (Optional[float  |  torch.nn.Module  |  Callable], optional): Dropout probability float, or a module. Defaults to None.
            act (Optional[torch.nn.Module  |  Callable], optional): _description_. Defaults to None.
            ndim (int, optional): Number of dimensions in input. Defaults to 2.
            order (str, optional): Order of operations by their first letters, 
            by default upscale -> convolution -> pooling -> activation -> norm -> dropout. Defaults to "UCPAND".
            device (_type_, optional): Something like `torch.device("cuda")`. Defaults to None.
            spatial_size (Optional[Sequence[int]], optional): Provide input spatial size, 
            sometimes needed if you use `LayerNorm` for example. Defaults to None.
            conv (torch.nn.Module | Callable | Literal[&#39;torch&#39;], optional): Module for convolution. 
            Defaults to 'torch' for ConvTransposend or LazyConvTransposend.
        """
        if scale != 2: raise ValueError(f'{scale = }, but ConvTranspose3StrideCube only supports scale = 2')

        kwargs = locals().copy()
        kwargs.pop('self')
        kwargs.pop('__class__')

        kwargs['kernel_size'] = 3
        kwargs['stride'] = 2
        kwargs['padding'] = 1
        kwargs['output_padding'] = 1
        kwargs.pop('scale')

        super().__init__(**kwargs)

    @classmethod
    def partial(cls, # type:ignore
        act: Optional[torch.nn.Module | Callable | str] = None,
        norm: Optional[torch.nn.Module | str | bool | Callable] = None,
        dropout: Optional[float | torch.nn.Module | Callable] = None,
        bias: bool | Literal['auto'] = 'auto',
        groups = 1,
        order = "UCPAND",
        device=None,
        spatial_size: Optional[Sequence[int]] = None,
        upconv: torch.nn.Module | Callable | Literal['torch'] = 'torch',
        ):

        kwargs = locals().copy()
        kwargs.pop('cls')

        return functools.partial(cls, **kwargs)


class _ConvTransposeCube(ConvTransposeBlock):
    """Convolution block"""
    def __init__(self,
        in_channels: Optional[int],
        out_channels: Optional[int] = None,
        scale: Optional[int] = None,
        resample: Optional[Literal["stride", "pool", "upsample"] | str | Callable | torch.nn.Module] = None,
        # padding: int | tuple = 0,
        # output_padding: int | tuple = 0,
        groups = 1,
        bias: bool | Literal['auto'] = 'auto',
        # dilation: int = 1,
        norm: Optional[torch.nn.Module | str | bool | Callable] = None,
        dropout: Optional[float | torch.nn.Module | Callable] = None,
        act: Optional[torch.nn.Module | Callable] = None,
        # upsample: Optional[float | torch.nn.Module | Callable] = None,
        # pool: Optional[int | torch.nn.Module | Callable] = None,
        ndim: int = 2,
        order = "UCPAND",
        device=None,
        spatial_size: Optional[Sequence[int]] = None,
        upconv: torch.nn.Module | Callable | Literal['torch'] = 'torch'
        ):
        """This is essentially a wrapper around ConvTransposeBlock that uses Cube interface, use ConvTransposeBlock directly if you need more sensible params.

        Args:
            in_channels (Optional[int]): Input channels, if `None` uses LazyConvnd
            out_channels (Optional[int], optional): Output channels, if `None` uses `input_channels`. Defaults to None.
            scale (Optional[float], optional): Input spatial size will be changed by this value (`kernel_size` and `stride` will be set to this). Defaults to None.
            groups (int, optional): _description_. Defaults to 1.
            bias (bool | Literal[&#39;auto&#39;], optional): On `auto`, disables bias if `norm` is not `None`. 
            Make sure to set this to `True` if you are using non-learnable norm. Defaults to 'auto'.
            norm (Optional[torch.nn.Module  |  str  |  bool  |  Callable], optional): _description_. Defaults to None.
            dropout (Optional[float  |  torch.nn.Module  |  Callable], optional): Dropout probability float, or a module. Defaults to None.
            act (Optional[torch.nn.Module  |  Callable], optional): _description_. Defaults to None.
            ndim (int, optional): Number of dimensions in input. Defaults to 2.
            order (str, optional): Order of operations by their first letters, 
            by default upscale -> convolution -> pooling -> activation -> norm -> dropout. Defaults to "UCPAND".
            device (_type_, optional): Something like `torch.device("cuda")`. Defaults to None.
            spatial_size (Optional[Sequence[int]], optional): Provide input spatial size, 
            sometimes needed if you use `LayerNorm` for example. Defaults to None.
            conv (torch.nn.Module | Callable | Literal[&#39;torch&#39;], optional): Module for convolution. 
            Defaults to 'torch' for ConvTransposend or LazyConvTransposend.
        """

        kwargs = locals().copy()
        kwargs.pop('self')

        if scale is None:
            ...
        kwargs['kernel_size'] = scale
        kwargs['stride'] = scale
        kwargs.pop('scale')

        super().__init__(**kwargs)

    @classmethod
    def partial(cls, # type:ignore
        groups = 1,
        bias: bool | Literal['auto'] = 'auto',
        dilation: int = 1,
        norm: Optional[torch.nn.Module | str | bool | Callable] = None,
        dropout: Optional[float | torch.nn.Module | Callable] = None,
        act: Optional[torch.nn.Module | Callable] = None,
        # upsample: Optional[float | torch.nn.Module | Callable] = None,
        # pool: Optional[int | torch.nn.Module | Callable] = None,
        ndim: int = 2,
        order = "UCPAND",
        device=None,
        spatial_size: Optional[Sequence[int]] = None,
        upconv: torch.nn.Module | Callable | Literal['torch'] = 'torch'
        ):

        kwargs = locals().copy()
        return _get_partial_from_locals(cls, kwargs)
