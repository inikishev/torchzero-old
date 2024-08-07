from collections.abc import Sequence
from typing import Literal
import torch
from ...python_tools import reduce_dim
from ..functional.pad import pad, pad_to_shape, pad_to_channels
__all__ = [
    "Pad",
    "PadToShape",
    "PadToChannels",
]


class Pad(torch.nn.Module):
    def __init__(self, padding:Sequence, mode='constant', value=None):
        super().__init__()
        self.padding = padding
        self.mode = mode
        self.value = value

    def forward(self, x:torch.Tensor):
        return pad(x, self.padding, self.mode, self.value)

class PadToShape(torch.nn.Module):
    def __init__(self, shape, where:Literal['center', 'start', 'end'] = 'center', mode='constant', value=None):
        """Pads `input` to have the same shape as `target`.

        Args:
            where (str, optional): How to pad, if `center`, input will be in the center, etc. Defaults to 'center'.
            mode (str, optional): Padding mode from `torch.nn.functional.pad`, can be 'constant', 'reflect', 'replicate' or 'circular'. Defaults to 'constant'.
            value (_type_, optional): Constant padding value if `mode` is `constant`. Defaults to None.
        """
        super().__init__()
        self.shape = shape
        self.where: Literal['center', 'start', 'end'] = where
        self.mode = mode
        self.value = value

    def forward(self, input:torch.Tensor):
        return pad_to_shape(input, shape=self.shape, where=self.where, mode=self.mode, value=self.value)


class PadToChannels(torch.nn.Module):
    def __init__(self, channels, where:Literal['center', 'start', 'end'] = 'center', mode='constant', value=None):
        """Pads `input` to have the same shape as `target`.

        Args:
            where (str, optional): How to pad, if `center`, input will be in the center, etc. Defaults to 'center'.
            mode (str, optional): Padding mode from `torch.nn.functional.pad`, can be 'constant', 'reflect', 'replicate' or 'circular'. Defaults to 'constant'.
            value (_type_, optional): Constant padding value if `mode` is `constant`. Defaults to None.
        """
        super().__init__()
        self.channels = channels
        self.where: Literal['center', 'start', 'end'] = where
        self.mode = mode
        self.value = value

    def forward(self, input:torch.Tensor):
        return pad_to_channels(input, channels=self.channels, where=self.where, mode=self.mode, value=self.value)
