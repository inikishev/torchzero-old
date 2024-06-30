from collections.abc import Sequence
import functools
import torch

from ..quick import seq
def _process_partial_seq(x):
    if callable(x): return x
    elif isinstance(x, Sequence):
        def partial_seq(in_channels, out_channels, scale, ndim):
            if out_channels is None: channels = [None] * (len(x) + 1)
            else: channels = [in_channels] * len(x) + [out_channels]
            if scale is None: scales = [None] * len(x)
            else: scales = [1] * (len(x)-1) + [scale]
            return seq([f(in_channels = channels[i], out_channels = channels[i+1], scale = scales[i], ndim=ndim) for i,f in enumerate(x)])
        return partial_seq
    else: raise ValueError(f'Invalid input type: {x}')
    
    
class _UnsupportedByThisCube: pass
_unsupported_by_this_cube = _UnsupportedByThisCube()


def _snap_int(i:int, snap:int):
    if i <= snap: return i
    diff = i % snap
    if diff < snap / 2: return int(i - diff)
    else: return int(i + (snap - diff))
    
    
class PartialIgnore: pass
def _ignore_PartialIgnores(kwargs:dict):
    return {k:v for k,v in kwargs.items() if not isinstance(v, PartialIgnore)}

def _get_partial_from_locals(cls, locals_copy:dict):
    locals_copy.pop('cls', None)
    locals_copy.pop('self', None)
    locals_copy.pop('__class__', None)
    return functools.partial(cls, **_ignore_PartialIgnores(locals_copy))