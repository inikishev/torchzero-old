"""Basic blocks"""
from typing import Optional, Literal
from collections.abc import Sequence, Callable
import functools

import numpy as np
import torch

from ..layers.func import ensure_module
from ..layers.pad import pad_like
from ._utils import _process_partial_seq, _unsupported_by_this_cube

__all__ = [
    'ResidualCube',
]

class ResidualCube(torch.nn.Module):
    def __init__(self, cube, in_channels, ndim = 2, out_channels = None, scale = None, resample=None):
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
        self.cube = _process_partial_seq(cube)(in_channels=in_channels, out_channels=out_channels, scale=scale, ndim = ndim)
        if resample is not None: self.resample = _process_partial_seq(resample)(in_channels=in_channels, out_channels=out_channels, scale=scale, ndim = ndim)
        else: self.resample = None

    def forward(self, x):
        out = self.cube(x)

        if self.resample is None:  return out + pad_like(out, x)

        resampled = self.resample(x)
        return out + pad_like(resampled, out)

    @classmethod
    def partial(cls, cube, resample = None):
        return functools.partial(cls, cube=cube, resample=resample)
