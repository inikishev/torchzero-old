"""Basic blocks"""
from typing import Optional, Literal
from collections.abc import Sequence, Callable
import functools
import torch
from ..layers.conv import ConvBlock
from ._utils import unsupported_by_this_cube, _get_partial_from_locals

__all__ = [
    "UpsampleCube",
]

class UpsampleCube(torch.nn.Upsample):
    def __init__(
        self,
        in_channels = unsupported_by_this_cube,
        out_channels = unsupported_by_this_cube,
        scale: int | tuple = 2,
        ndim = 2,
        mode: Literal['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'] = 'nearest',
        align_corners: bool | None = None,
        recompute_scale_factor: bool | None = None,
    ):
        super().__init__(scale_factor=scale, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
        if out_channels is not None and out_channels != in_channels:
            raise ValueError(f"UpsampleCube can't change number of channels, but {in_channels = }, and {out_channels = }")

    @classmethod
    def partial(cls, # type:ignore
        mode: Literal['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'] = 'nearest',
        align_corners: bool | None = None,
        recompute_scale_factor: bool | None = None,
        in_channels = unsupported_by_this_cube,
        out_channels = unsupported_by_this_cube,
        ):
        kwargs = locals().copy()
        return _get_partial_from_locals(cls, kwargs)
