from typing import Callable, Any, Optional
from collections.abc import Sequence
import functools
import torch

from ..quick import seq


CubePartial = Callable | Sequence[Callable] | Any


def _list_to_partial_seq(x: Sequence[Callable | Any]) -> Callable:
    x = list(x)
    if isinstance(x[-1], (int, tuple, dict)): x, indexes_ = x[:-1], x[-1]
    else: indexes_ = 0

    if isinstance(indexes_, int): indexes: dict[str, int] = dict(channels = indexes_, scale = indexes_)
    elif isinstance(indexes_, tuple): indexes: dict[str, int]  = dict(channels = indexes_[0], scale = indexes_[1])
    else: indexes: dict[str, int] = indexes_

    for key in 'c', 'ch', 'channel':
        if key in indexes: indexes['channels'] = indexes.pop(key)

    for key in 's', 'sc', 'sz', 'size':
        if key in indexes: indexes['scale'] = indexes.pop(key)

    channelidx = indexes.get('channels', 0)
    scaleidx = indexes.get('scale', 0)
    
    def partial_sequential(in_channels, out_channels, scale, ndim, **kwargs):
        channels:list[Any] = [None for _ in range(len(x) + 1)]
        channels[channelidx + 1] = out_channels
        cur = in_channels
        for i,v in enumerate(channels.copy()):
            if v is None: channels[i] = cur
            else: cur = out_channels

        scales:list[Any] = [None for _ in range(len(x))]
        scales[scaleidx] = scale

        return seq([f(in_channels = channels[i], out_channels = channels[i+1], scale = scales[i], ndim=ndim, **kwargs) for i,f in enumerate(x)])
    return partial_sequential

def partial_seq(x: CubePartial) -> Callable:
    if callable(x): return x
    elif isinstance(x, Sequence): return _list_to_partial_seq(x)
    else: raise ValueError(f'Invalid input type: {x}')


class _UnsupportedByThisCube: pass
unsupported_by_this_cube = _UnsupportedByThisCube()


def _snap_int(i:int, snap:int):
    if i <= snap: return i
    diff = i % snap
    if diff < snap / 2: return int(i - diff)
    else: return int(i + (snap - diff))


class _PartialIgnore: pass
partial_ignore = _PartialIgnore()

def _ignore_PartialIgnores(kwargs:dict):
    return {k:v for k,v in kwargs.items() if not isinstance(v, _PartialIgnore)}

def _get_partial_from_locals(cls, locals_copy:dict):
    locals_copy.pop('cls', None)
    locals_copy.pop('self', None)
    locals_copy.pop('__class__', None)
    return functools.partial(cls, **_ignore_PartialIgnores(locals_copy))
