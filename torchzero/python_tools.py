from collections.abc import Iterable
from typing import Any
import functools, operator
def reduce_dim(a: Iterable[Iterable]) -> list:
    return functools.reduce(operator.iconcat, a, []) # type:ignore

def flatten(iterable: Iterable) -> list:
    if isinstance(iterable, Iterable):
        return [a for i in iterable for a in flatten(i)]
    else:
        return [iterable]
