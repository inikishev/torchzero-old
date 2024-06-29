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
    'RecurrentCube',
]

class RecurrentCube(torch.nn.Module):
    def __init__(self, cube, in_channels:int, times:int, ndim = 2, scale = None, out_channels = _unsupported_by_this_cube):
        """_summary_

        Args:
            cube (_type_): _description_
            in_channels (int): _description_
            times (int): _description_
            ndim (int, optional): _description_. Defaults to 2.
            scale (_type_, optional): _description_. Defaults to None.
            out_channels (_type_, optional): _description_. Defaults to _unsupported_by_this_cube.
        """
        super().__init__()
        self.times = times
        self.cube = _process_partial_seq(cube)(in_channels=in_channels, out_channels=out_channels, scale=scale, ndim = ndim)

    def forward(self, x):
        for _ in range(self.times):
            x = self.cube(x)
        return x
    
    @classmethod
    def partial(cls, cube, times:int):
        return functools.partial(cls, cube=cube, times=times)