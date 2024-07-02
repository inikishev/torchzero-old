"""Basic blocks"""
from typing import Optional, Literal
from collections.abc import Sequence, Callable
import functools
import random

import numpy as np
import torch

from ..layers.func import ensure_module
from ..layers.pad import pad_like, pad_to_channels_like
from ._utils import partial_seq, unsupported_by_this_cube, _get_partial_from_locals, CubePartial

def _skip_cat(x, y): return torch.cat([x, y], dim=1)
def _skip_cat_post_sum(x:torch.Tensor):
    split = int(x.size(1) / 2)
    return x[:, :split] + x[:, split:]
def _skip_cat_post_mul(x:torch.Tensor):
    split = int(x.size(1) / 2)
    return x[:, :split] * x[:, split:]

def _skip_sum(x, y): return x + pad_to_channels_like(y, x)
def _skip_mul(x, y): return x * pad_to_channels_like(y, x)

def _skip_batch(x, y): return torch.cat([x, y], dim=0)
def _skip_batch_post_sum(x:torch.Tensor):
    split = int(x.size(0) / 2)
    return x[:split] + x[split:]
def _skip_batch_post_mul(x:torch.Tensor):
    split = int(x.size(0) / 2)
    return x[:split] * x[split:]
def _skip_batch_post_cat(x:torch.Tensor):
    split = int(x.size(0) / 2)
    return torch.cat([x[:split], x[split:]], dim=1)

def _skip_spatial(x, y): return torch.cat([x, pad_to_channels_like(y, x)], dim=2)
def _skip_spatial_post_sum(x:torch.Tensor):
    split = int(x.size(2) / 2)
    return x[:, :, :split] + x[:, :, split:]
def _skip_spatial_post_mul(x:torch.Tensor):
    split = int(x.size(2) / 2)
    return x[:, :, :split] * x[:, :, split:]
def _skip_spatial_post_cat(x:torch.Tensor):
    split = int(x.size(2) / 2)
    return torch.cat([x[:, :, :split], x[:, :, split:]], dim=1)

def _skip_dim(x, y): return torch.stack([x, pad_to_channels_like(y, x)], dim=2)
def _skip_dim_post_sum(x:torch.Tensor): return x[:,:,0] + x[:,:,1]
def _skip_dim_post_mul(x:torch.Tensor): return x[:,:,0] * x[:,:,1]
def _skip_dim_post_cat(x:torch.Tensor): return x.flatten(1,2)

def _skip_none(x, y): return x
def _skip_skip(x, y): return y

def _identity(x): return x

_SKIP_FNS: dict[str, tuple[Callable, Callable]] = dict(
    cat = (_skip_cat, _identity),
    catsum = (_skip_cat, _skip_cat_post_sum),
    catmul = (_skip_cat, _skip_batch_post_mul),
    sum = (_skip_sum, _identity),
    mul = (_skip_mul, _identity),
    batch = (_skip_batch, _identity),
    batchsum = (_skip_batch, _skip_batch_post_sum),
    batchmul = (_skip_batch, _skip_batch_post_mul),
    batchcat = (_skip_batch, _skip_batch_post_cat),
    spatial = (_skip_spatial, _identity),
    spatialsum = (_skip_spatial, _skip_spatial_post_sum),
    spatialmul = (_skip_spatial, _skip_spatial_post_mul),
    spatialcat = (_skip_spatial, _skip_spatial_post_cat),
    dim = (_skip_dim, _identity),
    dimsum = (_skip_dim, _skip_dim_post_sum),
    dimmul = (_skip_dim, _skip_dim_post_mul),
    dimcat = (_skip_dim, _skip_dim_post_cat),
    none = (_skip_none, _identity),
    skip = (_skip_skip, _identity),
)

SkipLiteral = Literal['cat', 'catsum', 'catmul', 'sum', 'mul', 'batch', 'batchsum', 'batchmul', 'batchcat', 'spatial', 'spatialsum', 'spatialmul', 'spatialcat', 'dim', 'dimsum', 'dimmul', 'dimcat', 'none', 'skip']

class SkipCube(torch.nn.Module):
    """Skip a cube."""
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels: Optional[int] = None,
        ndim=2,
        scale = None,
        cube: Optional[CubePartial] = None,
        skip: SkipLiteral = "cat",
        skip_cube: Optional[CubePartial] = None,
        skip_scale: Optional[float] = None,
        skip_out_channels: Optional[int] = None,
        orig_cube: Optional[CubePartial] = None,
        orig_scale: Optional[float] = None,
        orig_out_channels: Optional[int] = None,
        skip_dropout: Optional[float]  = None,
        orig_dropout: Optional[float]  = None,
        swap_p: Optional[float] = None,
    ):
        """Calculates
        ```js
        x, y -> postskip_fn(cube(skip_fn(x, y)))
        ```
        Or
        ```js
        x, y -> postskip_fn(
            cube(
                skip_fn(orig_cube(x), skip_cube(y))
                )
            )
        ```
        This will handle cubes in and out channels regardless of skip type.
        The partial may also need `skip_channels` argument for some skip modes.

        Args:
            cube (_type_): _description_
            in_channels (_type_): _description_
            out_channels (_type_): _description_
            skip_channels (_type_): _description_
            ndim (int, optional): _description_. Defaults to 2.
            scale (_type_, optional): _description_. Defaults to None.
            skip (Literal[&quot;cat&quot;, &quot;sum&quot;, &quot;batch&quot;, &quot;batchsum&quot;, &quot;batchmul&quot;, &quot;batchcat&quot;, &quot;spatial&quot;, &quot;spatialsum&quot;, &quot;spatialmul&quot;, &quot;spatialcat&quot;, &quot;dim&quot;, &quot;dimsum&quot;, &quot;dimmul&quot;, &quot;dimcat&quot;, &quot;none&quot;, &quot;skip&quot;], optional): _description_. Defaults to "cat".
            skip_dropout (_type_, optional): _description_. Defaults to None.
            orig_dropout (_type_, optional): _description_. Defaults to None.
        """
        super().__init__()

        if skip_cube is not None:
            self.skip_cube = partial_seq(skip_cube)(in_channels=skip_channels, out_channels=skip_out_channels, scale=skip_scale, ndim=ndim)
        else: self.skip_cube = _identity
        if skip_cube is not None and skip_out_channels is not None: skip_channels = skip_out_channels

        if orig_cube is not None:
            self.orig_cube = partial_seq(orig_cube)(in_channels=in_channels, out_channels=orig_out_channels, scale=orig_scale, ndim=ndim)
        else: self.orig_cube = _identity
        if orig_cube is not None and orig_out_channels is not None: in_channels = orig_out_channels

        if skip.startswith('cat'): in_channels += skip_channels # type:ignore
        if skip.startswith('dim'): ndim += 1

        if skip.endswith('cat') and skip not in ('cat',):
            if out_channels % 2 != 0: raise ValueError(f"out_channels must be divisible by 2 for skip modes that end with 'cat', but it is `{out_channels}`.")
            out_channels = int(out_channels / 2)

        if skip.startswith('cat') and skip not in ('cat',):
            out_channels *= 2
        
        if cube is not None: self.cube = partial_seq(cube)(in_channels=in_channels, out_channels=out_channels, scale=scale, ndim=ndim)
        else: self.cube = _identity

        self.skip_dropout = skip_dropout if skip_dropout is not None else 0
        self.orig_dropout = orig_dropout if orig_dropout is not None else 0
        self.swap_p = swap_p if swap_p is not None else 0

        self.skip = skip
        self.skip_fn, self.post_fn = _SKIP_FNS[skip]


    def forward(self, x, x_skip):
        skip_dropout = torch.rand(1) < self.skip_dropout
        orig_dropout = torch.rand(1) < self.orig_dropout
        swap = torch.rand(1) < self.swap_p
        if skip_dropout and orig_dropout:
            if torch.rand(1) < 0.5: skip_dropout = False
            else: orig_dropout = False

        if orig_dropout:
            if "mul" in self.skip: x = torch.ones_like(x)
            else: x = torch.zeros_like(x)
        if skip_dropout:
            if "mul" in self.skip: x_skip = torch.ones_like(x_skip)
            x_skip = torch.zeros_like(x_skip)

        if swap: x, x_skip = x_skip, x

        return self.post_fn(self.cube(self.skip_fn(self.orig_cube(x), self.skip_cube(x_skip))))


    @classmethod
    def partial(cls, # type:ignore
        cube,
        skip: Literal["cat", "sum", "batch", "batchsum", "batchmul", "batchcat", "spatial", "spatialsum", "spatialmul", "spatialcat", "dim", "dimsum", "dimmul", "dimcat", "none", "skip"] = "cat",
        skip_dropout = None,
        orig_dropout = None,
        ):
        kwargs = locals().copy()
        return _get_partial_from_locals(cls, kwargs)


