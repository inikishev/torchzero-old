"""Those are simply references for the class interfaces to make sure blocks are defined in a way net constructors can use them. I don't even inherit from them."""
from typing import Optional
from abc import ABC, abstractmethod
import functools

__all__ = ['CubeBase']\

class CubeBase(ABC): pass

class __Interface(ABC):
    """Changes channel number according to `out_channels` and changes spatial size according to `scale`."""
    @abstractmethod
    def __init__(self, in_channels: int, out_channels: Optional[int], scale: Optional[float], ndim: int, **kwargs): ...

    @abstractmethod
    @classmethod
    def partial(cls, **kwargs): return functools.partial(cls, **kwargs)