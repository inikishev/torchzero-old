"""Basic blocks"""
from typing import Optional, Literal
from collections.abc import Sequence, Callable
import functools
import torch
from ..layers.conv import ConvBlock
from .._library.pool import create_pool
from ._utils import _get_partial_from_locals

__all__ = [
    "ConvCube",
]
class ConvCube(ConvBlock):
    def __init__(self,
        in_channels: Optional[int],
        out_channels: Optional[int] = None,
        scale: Optional[float] = None,
        ndim: int = 2,
        kernel_size: int | tuple = 2,
        act: Optional[torch.nn.Module | Callable | str] = None,
        norm: Optional[torch.nn.Module | str | bool | Callable] = None,
        dropout: Optional[float | torch.nn.Module | Callable] = None,
        bias: bool | Literal['auto'] = 'auto',
        groups: int = 1,
        resample: Optional[Literal["stride", "pool", "upsample"] | str | Callable | torch.nn.Module] = None,
        order = "UCPAND",
        device = None,
        spatial_size: Optional[Sequence[int]] = None,
        conv: torch.nn.Module | Callable | Literal['torch'] = 'torch',
        # stride: int | tuple  = 1,
        # padding: int | tuple | Literal['auto'] = 'auto',
        # dilation: int | tuple = 1,
        # pool: Optional[int | torch.nn.Module | Callable] = None,
        ):
        """This is essentially a wrapper around ConvBlock that uses Cube interface, use ConvBlock directly if you need more sensible params.

        Args:
            in_channels (Optional[int]): Input channels, if `None` uses LazyConvnd
            out_channels (Optional[int], optional): Output channels, if `None` uses `input_channels`. Defaults to None.
            scale (Optional[float], optional): Input spatial size will be changed by this value, if not `None`. Defaults to None.
            resample: Module for resampling, only used if `scale` is not None. Can be stride, pooling, upsample, etc. Defaults to None.
            kernel_size (int | tuple, optional): Kernel size. Padding will be adjusted automatically based on this. Defaults to 2.
            groups (int, optional): _description_. Defaults to 1.
            bias (bool | Literal[&#39;auto&#39;], optional): On `auto`, disables bias if `norm` is not `None`. 
            Make sure to set this to `True` if you are using non-learnable norm. Defaults to 'auto'.
            norm (Optional[torch.nn.Module  |  str  |  bool  |  Callable], optional): _description_. Defaults to None.
            dropout (Optional[float  |  torch.nn.Module  |  Callable], optional): Dropout probability float, or a module. Defaults to None.
            act (Optional[torch.nn.Module  |  Callable], optional): _description_. Defaults to None.
            residual (bool, optional): Adds input to output, don't use with scale other than `None` / `1`. Defaults to False.
            recurrent: Applies this block (including pooling or upsample) this many times. 
            Make sure `out_channels` isn't different from `in_channels`. Defaults to 1.
            ndim (int, optional): Number of dimensions in input. Defaults to 2.
            order (str, optional): Order of operations by their first letters, 
            by default upscale -> convolution -> pooling -> activation -> norm -> dropout. Defaults to "UCPAND".
            device (_type_, optional): Something like `torch.device("cuda")`. Defaults to None.
            spatial_size (Optional[Sequence[int]], optional): Provide input spatial size, 
            sometimes needed if you use `LayerNorm` for example. Defaults to None.
            conv (torch.nn.Module | Callable | Literal[&#39;torch&#39;], optional): Module for convolution. 
            Defaults to 'torch' for Convnd or LazyConvnd.

        Raises:
            ValueError: _description_
        """

        kwargs = locals().copy()
        kwargs.pop('self')
        kwargs.pop('__class__')
        
        # no rescaling
        if scale is None: kwargs["stride"] = 1
        else:
            # if resample is None
            if resample is None:
                if scale <= 1: resample = 'stride'
                else: resample = 'upsample'

            # if resample is callable, stride is one and resample is considered the pool module
            if callable(resample):
                kwargs["stride"] = 1
                kwargs["pool"] = resample
            # if resample is stride, we set stride to the required value
            elif resample == 'stride':
                if (1 / scale) % 1 != 0: raise ValueError(f"1/scale must be an integer for stride resampling, but {1/scale = }.")
                kwargs['stride'] = int(1 / scale)
            # if resample is upsample, set stride to one and set upsample to the value (uses billinear upsample)
            elif resample == 'upsample':
                kwargs["stride"] = 1
                kwargs['upsample'] = scale
            # if resample is either `pool` or some other string,
            # set stride to one and create a pool from the string given `scale`,
            # which is why we create it at this stage and don't just pass a string.
            else:
                kwargs['stride'] = 1
                kwargs['pool'] = create_pool(module=resample, num_channels=out_channels, ndim=ndim, spatial_size=spatial_size, scale=scale)

        kwargs.pop('scale')
        kwargs.pop('resample')
        
        super().__init__(**kwargs)

    @classmethod
    def partial(cls, # type:ignore
        kernel_size: int | tuple = 2,
        act: Optional[torch.nn.Module | Callable | str] = None,
        norm: Optional[torch.nn.Module | str | bool | Callable] = None,
        dropout: Optional[float | torch.nn.Module | Callable] = None,
        bias: bool | Literal['auto'] = 'auto',
        groups: int = 1,
        resample: Optional[Literal["stride", "pool", "upsample"] | str | Callable | torch.nn.Module] = None,
        order = "UCPAND",
        device = None,
        spatial_size: Optional[Sequence[int]] = None,
        conv: torch.nn.Module | Callable | Literal['torch'] = 'torch',
        ):

        kwargs = locals().copy()
        return _get_partial_from_locals(cls, kwargs)

