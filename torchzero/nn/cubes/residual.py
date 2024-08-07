"""Basic blocks"""
from typing import Optional, Literal
from collections.abc import Sequence, Callable
import functools

import numpy as np
import torch

from ..layers.func import ensure_module
from ..functional.pad import pad_like
from .skip import SkipCube, SkipLiteral
from ._utils import partial_seq, unsupported_by_this_cube

__all__ = [
    'ResidualCube',
]

class ResidualCube(torch.nn.Module):
    def __init__(
        self,
        cube,
        in_channels,
        ndim=2,
        out_channels=None,
        scale=None,
        resample=None,
        skip_mode: SkipLiteral = "sum",
    ):
        """Basic residual block with identity connection.

        Args:
            cube (_type_): Cube to use.
            in_channels (_type_): input channels
            ndim (int, optional): number of dims. Defaults to 2.
            out_channels (_type_, optional): out channels, should be same or bigger than in channels. Defaults to None.
            scale (_type_, optional): Scale, if set, `resample` must be set as well. Defaults to None.
            resample (_type_, optional): Cube that will resample input to the same size as main cube. Defaults to None.
        """
        super().__init__()
        self.cube = partial_seq(cube)(in_channels = in_channels, out_channels = out_channels, scale = scale, ndim = ndim)
        self.skip = SkipCube(
            in_channels = out_channels,
            out_channels = out_channels,
            skip_channels = in_channels,
            ndim=ndim,
            scale=scale,
            cube=None,
            skip_mode = skip_mode,
            skip_cube = resample,
            skip_scale = scale,
        )

    def forward(self, x):
        out = self.cube(x)
        return self.skip(out, x)

    @classmethod
    def partial(cls, cube, resample = None, skip_mode: SkipLiteral = 'sum'):
        return functools.partial(cls, cube=cube, resample=resample, skip_mode = skip_mode)
