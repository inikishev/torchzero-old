from collections.abc import Callable, Sequence, Iterable
from typing import Any, Optional
import functools, operator
def reduce_dim(a: Iterable[Iterable]) -> list:
    """ultra fast function that flattens any iterable by one dimension, from there https://stackoverflow.com/a/45323085/15673832"""
    return functools.reduce(operator.iconcat, a, []) # type:ignore

def flatten(iterable: Iterable) -> list:
    if isinstance(iterable, Iterable):
        return [a for i in iterable for a in flatten(i)]
    else:
        return [iterable]

def identity(x): return x

class Compose:
    """Compose"""
    def __init__(self, *transforms):
        self.transforms = flatten(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __add__(self, other: "Compose | Callable | Iterable"):
            return Compose(*self.transforms, other)

    def __str__(self):
        return f"Compose({', '.join(str(t) for t in self.transforms)})"

    def __iter__(self):
        return iter(self.transforms)

    def __getitem__(self, i): return self.transforms[i]
    def __setitem__(self, i, v): self.transforms[i] = v
    def __delitem__(self, i): del self.transforms[i]

def auto_compose(func: Optional[Callable | Sequence[Callable]]):
    """Composes `func` if it is a sequence, returns `identity` if it is None.

    Args:
        func (Optional[Callable  |  Sequence[Callable]]): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(func, Sequence): return Compose(*func)
    if func is None: return identity
    return func