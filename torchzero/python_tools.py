from collections.abc import Iterable
from typing import Any
import functools, operator
def reduce_dim(a: Iterable[Iterable]) -> list:
    """ultra fast function that flattens any iterable by one dimension, from there https://stackoverflow.com/a/45323085/15673832"""
    return functools.reduce(operator.iconcat, a, []) # type:ignore

def flatten(iterable: Iterable) -> list:
    if isinstance(iterable, Iterable):
        return [a for i in iterable for a in flatten(i)]
    else:
        return [iterable]
