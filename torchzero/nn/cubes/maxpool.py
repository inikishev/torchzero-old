"""Basic blocks"""
from typing import Optional, Literal
from collections.abc import Sequence, Callable
import functools
import torch
from ..layers.conv import ConvBlock
from .._library.pool import create_pool, get_maxpoolnd
from ._utils import _unsupported_by_this_cube

__all__ = [
    "MaxPoolCube",
]

class MaxPoolCube(torch.nn.Module):
    def __init__(
        self,
        scale: float = 0.5,
        ndim = 2,
        # kernel_size: int | tuple,
        # stride: int | tuple | None = None,
        # padding: int | tuple = 0,
        # dilation: int | tuple = 1,
        # return_indices: bool = False,
        ceil_mode: bool = False,

        in_channels = _unsupported_by_this_cube,
        out_channels = _unsupported_by_this_cube,
    ):
        super().__init__()
        if out_channels is not None and out_channels != in_channels:
            raise ValueError(f"MaxPoolCube can't change number of channels, but {in_channels = }, and {out_channels = }")

        self.maxpool = get_maxpoolnd(ndim)(kernel_size=int(1/scale), stride=int(1/scale), ceil_mode=ceil_mode)

    def forward(self, x:torch.Tensor):
        return self.maxpool(x)

    @classmethod
    def partial(cls, # type:ignore
        ceil_mode: bool = False,
        ):

        return functools.partial(cls, ceil_mode = ceil_mode)
