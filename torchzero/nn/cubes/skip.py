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

__all__ = [
    "SkipCube",
    "PassAndSkipCube",
    "SaveCube",
]
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
        scale = None,
        ndim=2,
        #
        cube: Optional[CubePartial] = None,
        skip_mode: SkipLiteral = "cat",
        skip_cube: Optional[CubePartial] = None,
        skip_scale: Optional[float] = None,
        skip_out_channels: Optional[int] = None,
        orig_cube: Optional[CubePartial] = None,
        orig_scale: Optional[float] = None,
        orig_out_channels: Optional[int] = None,
        post_cube: Optional[CubePartial] = None,
        post_scale: Optional[float] = None,
        post_in_channels: Optional[int] | Literal['skip'] = None,
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
        The partial may also need `skip_channels` argument for some skip modes

        Args:
            in_channels (_type_): _description_
            out_channels (_type_): _description_
            skip_channels (Optional[int], optional): How many input channels in `x_skip`. Defaults to None.
            scale (_type_, optional): Scale that gets passed to `cube`, doesn't affect `orig_cube`, `skip_cube`, `post_cube`. Defaults to None.
            ndim (int, optional): _description_. Defaults to 2.
            cube (Optional[CubePartial], optional): Cube that gets applied to `x` after skipping. Defaults to None.
            skip_mode (SkipLiteral, optional): _description_. Defaults to "cat".
            skip_cube (Optional[CubePartial], optional): _description_. Defaults to None.
            skip_scale (Optional[float], optional): _description_. Defaults to None.
            skip_out_channels (Optional[int], optional): How many output channels for skip cube, automatically used for `skip_channels`. Defaults to None.
            orig_cube (Optional[CubePartial], optional): _description_. Defaults to None.
            orig_scale (Optional[float], optional): _description_. Defaults to None.
            orig_out_channels (Optional[int], optional): How any output channels for orig cube, automaticaly used for `in_channels`. Defaults to None.
            post_cube (Optional[CubePartial], optional): _description_. Defaults to None.
            post_scale (Optional[float], optional): _description_. Defaults to None.
            post_in_channels (Optional[int], optional): Passed as `out_channels` to `cube`, and `post_cube` always sets `out_channels`. Defaults to None.
            skip_dropout (Optional[float], optional): _description_. Defaults to None.
            orig_dropout (Optional[float], optional): _description_. Defaults to None.
            swap_p (Optional[float], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        super().__init__()

        if skip_cube is not None:
            self.skip_cube = partial_seq(skip_cube)(in_channels=skip_channels, out_channels=skip_out_channels, scale=skip_scale, ndim=ndim)
        elif skip_out_channels is not None and skip_out_channels != skip_channels: raise ValueError(f"`skip_cube` is None, but {skip_out_channels = }; {skip_scale = }")
        elif skip_scale is not None and skip_scale != 1: raise ValueError(f"`skip_cube` is None, but {skip_out_channels = }; {skip_scale = }")
        else: self.skip_cube = _identity
        if skip_cube is not None and skip_out_channels is not None: skip_channels = skip_out_channels

        if orig_cube is not None:
            self.orig_cube = partial_seq(orig_cube)(in_channels=in_channels, out_channels=orig_out_channels, scale=orig_scale, ndim=ndim)
        elif orig_out_channels is not None and orig_out_channels != in_channels: raise ValueError(f"`orig_cube` is None, but {orig_out_channels = }; {orig_scale = }")
        elif orig_scale is not None and orig_scale != 1: raise ValueError(f"`orig_cube` is None, but {orig_out_channels = }; {orig_scale = }")
        else: self.orig_cube = _identity
        if orig_cube is not None and orig_out_channels is not None: in_channels = orig_out_channels

        if skip_mode.startswith('cat'): in_channels += skip_channels # type:ignore
        if skip_mode.startswith('dim'): ndim += 1

        if skip_mode.endswith('cat') and skip_mode not in ('cat',):
            if out_channels % 2 != 0: raise ValueError(f"out_channels must be divisible by 2 for skip modes that end with 'cat', but it is `{out_channels}`.")
            out_channels = int(out_channels / 2)

        if skip_mode.startswith('cat') and skip_mode not in ('cat',):
            out_channels *= 2

        cube_out_channels = out_channels if post_in_channels is None else post_in_channels
        if cube_out_channels == 'skip': cube_out_channels = in_channels
        if cube is not None: self.cube = partial_seq(cube)(in_channels=in_channels, out_channels=cube_out_channels, scale=scale, ndim=ndim)
        else:
            cube_out_channels = in_channels # pass that to `post_cube`
            self.cube = _identity

        self.skip_dropout = skip_dropout if skip_dropout is not None else 0
        self.orig_dropout = orig_dropout if orig_dropout is not None else 0
        self.swap_p = swap_p if swap_p is not None else 0

        self.skip = skip_mode
        self.skip_fn, self.post_fn = _SKIP_FNS[skip_mode]

        if post_cube is not None:
            self.post_cube = partial_seq(post_cube)(in_channels=cube_out_channels, out_channels=out_channels, scale=post_scale, ndim=ndim)
        elif post_in_channels is not None and post_in_channels != out_channels: raise ValueError(f"`post_cube` is None, but {post_in_channels = }; {post_scale = }")
        elif post_scale is not None and post_scale != 1: raise ValueError(f"`post_cube` is None, but {post_in_channels = }; {post_scale = }")
        else: self.post_cube = _identity

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

        return self.post_cube(self.post_fn(self.cube(self.skip_fn(self.orig_cube(x), self.skip_cube(x_skip)))))


    @classmethod
    def partial(cls, # type:ignore
        cube: Optional[CubePartial] = None,
        skip_mode: SkipLiteral = "cat",
        skip_cube: Optional[CubePartial] = None,
        skip_scale: Optional[float] = None,
        skip_out_channels: Optional[int] = None,
        orig_cube: Optional[CubePartial] = None,
        orig_scale: Optional[float] = None,
        orig_out_channels: Optional[int] = None,
        post_cube: Optional[CubePartial] = None,
        post_scale: Optional[float] = None,
        post_in_channels: Optional[int] = None,
        skip_dropout: Optional[float]  = None,
        orig_dropout: Optional[float]  = None,
        swap_p: Optional[float] = None,
        ):
        kwargs = locals().copy()
        return _get_partial_from_locals(cls, kwargs)



class PassAndSkipCube(SkipCube):
    def __init__(self,
                 in_channels,
                 out_channels,
                 skip_channels: Optional[int] = None,
                 scale = None,
                 ndim=2,
                 #
                 skip_mode: SkipLiteral = "cat",
                 pre_cube: Optional[CubePartial] = None,
                 cube: Optional[CubePartial] = None,
                 post_cube: Optional[CubePartial] = None,
                 skip_cube: Optional[CubePartial] = None,
                 skip_scale: Optional[float] = None,
                 skip_out_channels: Optional[int] = None,
                 skip_dropout: Optional[float]  = None,
                 orig_dropout: Optional[float]  = None,
                 swap_p: Optional[float] = None,
                 scale_module:Literal['pre', 'cube', 'post'] = 'pre',
                 channels_module:Literal['pre', 'cube', 'post'] = 'pre',
                 ):

        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            skip_channels = skip_channels,
            scale = scale if scale_module == 'cube' else None,
            ndim = ndim,
            #
            cube = cube,
            skip_mode = skip_mode,
            skip_cube = skip_cube,
            skip_scale = skip_scale,
            skip_out_channels = skip_out_channels,
            orig_cube = pre_cube,
            orig_scale = scale if scale_module == 'pre' else None,
            orig_out_channels = out_channels if channels_module == 'pre' else None,
            post_cube = post_cube,
            post_scale = scale if scale_module == 'post' else None,
            post_in_channels = 'skip' if channels_module == 'post' else None,
            skip_dropout = skip_dropout,
            orig_dropout = orig_dropout,
            swap_p = swap_p,
            )

    @classmethod
    def partial(cls, # type:ignore
            skip_mode: SkipLiteral = "cat",
            pre_cube: Optional[CubePartial] = None,
            cube: Optional[CubePartial] = None,
            post_cube: Optional[CubePartial] = None,
            skip_cube: Optional[CubePartial] = None,
            skip_scale: Optional[float] = None,
            skip_out_channels: Optional[int] = None,
            skip_dropout: Optional[float]  = None,
            orig_dropout: Optional[float]  = None,
            swap_p: Optional[float] = None,
            scale_module:Literal['pre', 'cube', 'post'] = 'pre',
            channels_module:Literal['pre', 'cube', 'post'] = 'pre',
            ):
        kwargs = locals().copy()
        return _get_partial_from_locals(cls, kwargs)

class SaveCube(torch.nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                scale = None,
                ndim=2,
                #
                pre_cube: Optional[CubePartial] = None,
                post_cube: Optional[CubePartial] = None,
                save_cube: Optional[CubePartial] = None,
                save_scale: Optional[float] = None,
                save_out_channels: Optional[int] = None,
                scale_module:Literal['pre', 'post'] = 'pre',
                channels_module:Literal['pre', 'post'] = 'pre',
                ):
        super().__init__()
        if pre_cube is not None:
            self.pre_cube = partial_seq(pre_cube)(
                in_channels=in_channels,
                out_channels=out_channels if channels_module == 'pre' else None,
                scale=scale if scale_module == 'pre' else None,
                ndim=ndim,
            )
        elif scale_module == 'pre' and scale is not None: raise ValueError(f"`pre_cube` is None, but {scale = } and {scale_module = }")
        elif channels_module == 'pre' and (out_channels is not None and out_channels != in_channels):
            raise ValueError(f"`pre_cube` is None, but {out_channels = } and {channels_module = }")
        else: self.pre_cube = _identity

        if post_cube is not None:
            self.post_cube = partial_seq(post_cube)(
                in_channels=out_channels if channels_module == 'pre' else in_channels,
                out_channels=out_channels,
                scale=scale if scale_module == 'post' else None,
                ndim=ndim,
            )
        elif scale_module == 'post' and scale is not None: raise ValueError(f"`post_cube` is None, but {scale = } and {scale_module = }")
        elif channels_module == 'post' and (out_channels is not None and out_channels != in_channels):
            raise ValueError(f"`post_cube` is None, but {out_channels = } and {channels_module = }")
        else: self.post_cube = _identity

        if save_cube is not None:
            self.save_cube = partial_seq(save_cube)(
                in_channels=out_channels if channels_module == 'pre' else in_channels,
                out_channels=save_out_channels,
                scale=save_scale,
                ndim=ndim,
            )
        else: save_cube = _identity

    def forward(self, x:torch.Tensor):
        """Returns `x`, `saved`."""
        pre = self.pre_cube(x)
        return self.post_cube(pre), self.save_cube(pre)

    @classmethod
    def partial(cls, # type:ignore
                pre_cube: Optional[CubePartial] = None,
                post_cube: Optional[CubePartial] = None,
                save_cube: Optional[CubePartial] = None,
                save_scale: Optional[float] = None,
                save_out_channels: Optional[int] = None,
                scale_module:Literal['pre', 'post'] = 'pre',
                channels_module:Literal['pre', 'post'] = 'pre',
            ):
        kwargs = locals().copy()
        return _get_partial_from_locals(cls, kwargs)