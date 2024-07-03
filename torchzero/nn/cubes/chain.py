"""Basic blocks"""
from typing import Optional, Literal
from collections.abc import Sequence, Callable, Mapping
import functools

import numpy as np
import torch

from ..layers.func import ensure_module
from ..layers.pad import pad_like
from ._utils import partial_seq, _snap_int, _get_partial_from_locals, CubePartial, unsupported_by_this_cube

__all__ = [
    'ChainCube',
    'StraightAndResizeCube',
    "SequenceCube",
]

def _generate_channels(in_channels, out_channels, mode, num, snap) -> list[int]:
    if out_channels is None: return [None] * (num + 1)
    if in_channels == out_channels: return [in_channels] * (num + 1)
    if isinstance(mode, (list, tuple)): return list(mode)

    if isinstance(mode, int):
        channels = [None for _ in range(num + 1)]
        channels[mode] = out_channels
        cur = in_channels
        for i in range(len(channels)):
            if channels[i] is None: channels[i] = cur
            else: cur = out_channels

    if mode == 'min': return [in_channels] + [min(in_channels, out_channels)] * (num - 1) + [out_channels]
    if mode == 'max': return [in_channels] + [max(in_channels, out_channels)] * (num - 1) + [out_channels]
    if mode == 'in': return [in_channels] * num + [out_channels]
    if mode == 'out': return [in_channels] + [out_channels] * num

    if mode == 'gradual': channels = np.linspace(in_channels, out_channels, num + 1, dtype=int).tolist()
    elif mode == 'mean': channels = [in_channels] + [int((in_channels + out_channels) / 2)] * (num - 1) + [out_channels]
    else: raise ValueError(f"Invalid channels_mode: {mode}")

    if snap is not None: channels = [channels[0]] + [_snap_int(i, snap) for i in channels[1:-1]] + [channels[-1]]
    return channels

class ChainCube(torch.nn.Sequential):
    def __init__(self,
        cube,
        num:int,
        in_channels: int,
        out_channels: Optional[int] = None,
        scale: Optional[float] = None,
        ndim = 2,
        channel_mode: Literal['in', 'out', 'min', 'max', 'mean', 'gradual'] | Sequence[int] | int = 'gradual',
        scale_cubes: Literal['first', 'last', 'all'] | Sequence[int] | int = 'all',
        snap_channels:Optional[int] = None,
        overrides: Optional[Mapping[int | slice, CubePartial]] = None,
    ):
        """Chains cubes.

        Args:
            cube (_type_): The cube that will be chained. Pass something like `ConvCube.partial(...)`.
            num (int): Number of cubes to chain.
            in_channels (int): Input channels.
            out_channels (Optional[int], optional): Output channels, if `None` uses `input_channels`. Defaults to None.
            channel_mode: How to increase/decrease channels per each cube. Defaults to 'gradual'.
            scale (Optional[float], optional): Output scale. Defaults to None.
            scale_cubes: Which cubes should increase/decrease scale, first, last, or all. Defaults to 'all'.
            residual (bool, optional): Adds input to output, don't use with scale other than `None` / `1`. Defaults to False.
            recurrent: Applies this block (including pooling or upsample) this many times.
            Make sure `out_channels` isn't different from `in_channels`. Defaults to 1.

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        cube = partial_seq(cube)
        cubes = [cube for _ in range(num)]

        if overrides is None: overrides = {}
        else: overrides = {k : partial_seq(v) for k,v in overrides.items()}

        for k,v in overrides.items():
            if isinstance(k, slice):
                for i in range(*k.indices(num)): cubes[i] = partial_seq(v)
            else: cubes[k] = partial_seq(v)

        self.channels = _generate_channels(in_channels, out_channels, mode=channel_mode, num=num, snap=snap_channels)

        if scale is None: scales = [None] * num
        elif scale_cubes == 'first': scales = [scale] + [1] * (num - 1)
        elif scale_cubes == 'last': scales = [1] * (num - 1) + [scale]
        elif scale_cubes == 'all': scales = [scale] * num
        elif isinstance(scale_cubes, Sequence): scales = scale_cubes
        else: raise ValueError(f"Invalid scale_cubes: {scale_cubes}")


        blocks = []
        for i, cube_ in enumerate(cubes):
            blocks.append(
                ensure_module(
                    cube_(
                        in_channels=self.channels[i],
                        out_channels=self.channels[i + 1],
                        scale=scales[i],
                        ndim=ndim,
                    )
                )
            )

        super().__init__(*blocks)



    @classmethod
    def partial(cls, # type:ignore
        cube: CubePartial,
        num: int,
        channel_mode: Literal['in', 'out', 'min', 'max', 'mean', 'gradual'] | Sequence[int] = 'gradual',
        scale_cubes: Literal['first', 'last', 'all'] | Sequence[int] = 'all',
        snap_channels:Optional[int] = None,
        overrides: Optional[Mapping[int | slice, CubePartial]] = None,
        ):
        kwargs = locals().copy()
        return _get_partial_from_locals(cls, kwargs)



class StraightAndResizeCube(torch.nn.Sequential):
    def __init__(self,
        straight_cube,
        resize_cube,
        straight_num:int,
        scale: float,
        in_channels: int,
        out_channels: Optional[int] = None,
        ndim = 2,
        channel_mode: Literal['in', 'out', 'min', 'max', 'mean', 'gradual'] | Sequence[int] = 'gradual',
        only_straight_channels = True,
        order: Literal['SR', 'RS'] = "SR",
        snap_channels:Optional[int] = None,
        straight_overrides: Optional[Mapping[int | slice, CubePartial]] = None,

    ):
        """Chains `straight_num` of `straight_cube`s, followed/preceded by an upsample/downsample `resize_cube`.

        Args:
            straight_cube (_type_): Straight cubes to chain. Pass something like `ConvCube.partial(...)`.
            resize_cube (_type_): Resize cube to upsample/downsample. E.g. `ConvCube.partial(resample='stride')` or `MaxPoolCube.partial()`.
            straight_num (int): Number of straight cubes to chain.
            scale (Optional[float]): Scale to pass to `resize_cube`.
            in_channels (int): Input channels.
            out_channels (Optional[int], optional): Output channels, if `None` uses `input_channels`. Defaults to None.
            channel_mode: How to increase/decrease channels per each cube. Defaults to 'gradual'.
            only_straight_channels: If `True`, only `straight_cube`s can change channel number.
            If `False`, `resize_cube` will also change channels number. Defaults to True.
            order: `SR` for straight -> resample, `RS` for resample -> straight. Defaults to "SR".

        Raises:
            ValueError: _description_
        """
        super().__init__()
        straight_cube = partial_seq(straight_cube)
        resize_cube = partial_seq(resize_cube)

        straight_cubes = [straight_cube for _ in range(straight_num)]

        if straight_overrides is None: straight_overrides = {}
        else: straight_overrides = {k:partial_seq(v) for k,v in straight_overrides.items()}

        for k,v in straight_overrides.items():
            if isinstance(k, slice):
                for i in range(*k.indices(straight_num)): straight_cubes[i] = partial_seq(v)
            else: straight_cubes[k] = partial_seq(v)

        num = straight_num
        if not only_straight_channels: num += 1

        self.channels = _generate_channels(in_channels, out_channels, mode=channel_mode, num=num, snap=snap_channels)

        blocks = []
        if order == 'SR':
                for i, straight_cube_ in enumerate(straight_cubes):
                    blocks.append(ensure_module(straight_cube_(in_channels = self.channels[i], out_channels = self.channels[i+1], scale=None, ndim=ndim)))
                if only_straight_channels:
                    blocks.append(ensure_module(resize_cube(in_channels = self.channels[-1], out_channels = None, scale=scale, ndim=ndim)))
                else:
                    blocks.append(ensure_module(resize_cube(in_channels = self.channels[-2], out_channels = self.channels[-1], scale=scale, ndim=ndim)))
        elif order == 'RS':
            if only_straight_channels:
                blocks.append(ensure_module(resize_cube(in_channels = self.channels[0], out_channels = None, scale=scale, ndim=ndim)))
            else:
                blocks.append(ensure_module(resize_cube(in_channels = self.channels[0], out_channels = self.channels[1], scale=scale, ndim=ndim)))
            for i, straight_cube_ in enumerate(straight_cubes):
                index = i + int(not only_straight_channels)
                blocks.append(ensure_module(straight_cube_(in_channels = self.channels[index], out_channels = self.channels[index+1], scale=None, ndim=ndim)))

        super().__init__(*blocks)

    @classmethod
    def partial(cls, # type:ignore
        straight_cube,
        resize_cube,
        straight_num:int,
        channel_mode: Literal['in', 'out', 'min', 'max', 'mean', 'gradual'] | Sequence[int] = 'gradual',
        only_straight_channels = True,
        order: Literal['SR', 'RS'] = "SR",
        snap_channels:Optional[int] = None,
        straight_overrides: Optional[Mapping[int | slice, CubePartial]] = None,
        ):
        kwargs = locals().copy()
        return _get_partial_from_locals(cls, kwargs)

class SequenceCube(torch.nn.Sequential):
    def __init__(self,
                 cubes:Sequence[CubePartial],
                 in_channels: int,
                 mid_channels: int | None | Sequence[int | None],
                 scale_idx: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 scale: Optional[float] = None,
                 ndim = 2,
                 ):
        """Chains a sequence of cubes.

        Args:
            cubes (Sequence[CubePartial]): Sequence of cubes to chain.
            in_channels (int): Input channels.
            out_channels (Optional[int], optional): Output channels, if `None` uses `input_channels`. Defaults to None.
            channel_mode: How to increase/decrease channels per each cube. Defaults to 'gradual'.
            scale (Optional[float]): Scale to pass to each cube.
            ndim (int, optional): Number of dimensions. Defaults to 2.

        Raises:
            ValueError: _description_
        """

        if isinstance(mid_channels, int) or mid_channels is None: mid_channels = [mid_channels for _ in range(len(cubes) - 1)]
        channels = [in_channels] + list(mid_channels) + [out_channels]
        if len(channels) != len(cubes) + 1: raise ValueError(f'{channels = } while {len(cubes) = }')

        scales: list[float | None] = [None] * len(cubes)
        if scale_idx is None and scale is not None: raise ValueError("passed scale to SequenceCube but scale_idx is None")
        if scale_idx is not None: scales[scale_idx] = scale

        cubes = [ensure_module(partial_seq(cube)(
            in_channels=channels[i],
            out_channels=channels[i+1],
            scale=scales[i],
            ndim=ndim,
            )) for i, cube in enumerate(cubes)]

        super().__init__(*cubes)


    @classmethod
    def partial(cls, # type:ignore
        cubes:Sequence[CubePartial],
        mid_channels: int | None | Sequence[int | None],
        scale_idx: Optional[int] = None,
        ):
        kwargs = locals().copy()
        return _get_partial_from_locals(cls, kwargs)