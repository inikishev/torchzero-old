from typing import Literal
import torch
def pad_to_shape(input:torch.Tensor, shape:tuple,
                 where:Literal['center', 'start', 'end'] = 'start',
                 mode='constant', value=None):
    """Pads `input` to have the same shape as `target`.

    Args:
        input (torch.Tensor): Input to pad.
        target (torch.Tensor): Target that the input will match the shape of, must have same or bigger shape among all dimensions.
        where (str, optional): How to pad, if `center`, input will be in the center, etc. Defaults to 'center'.
        mode (str, optional): Padding mode from `torch.nn.functional.pad`, can be 'constant', 'reflect', 'replicate' or 'circular'. Defaults to 'constant'.
        value (_type_, optional): Constant padding value if `mode` is `constant`. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if tuple(input.size()) == shape: return input
    padding = []
    for dim in range(len(shape)):
        diff = shape[dim] - input.size(dim)
        if diff > 0:
            if diff % 2 == 0:
                if where == 'center': padding.extend([int(diff / 2)] * 2)
                elif where == 'start': padding.extend([0, int(diff)])
                elif where == 'end': padding.extend([int(diff), 0])
                else: raise ValueError(f'Invalid where: {where}')
            else:
                if diff == 1:
                    if where in ('center', 'end'): padding.extend([1, 0])
                    elif where == 'start': padding.extend([0, 1])
                    else: raise ValueError(f'Invalid where: {where}')
                else:
                    if where == 'center': padding.extend([int(diff // 2), int(diff // 2) + 1])
                    elif where == 'start': padding.extend([0, int(diff)])
                    elif where == 'end': padding.extend([int(diff), 0])

        else: padding.extend([0,0])
    return torch.nn.functional.pad(input, list(reversed(padding)), mode=mode, value=value)

def pad_like(input:torch.Tensor, target:torch.Tensor, where:Literal['center', 'start', 'end'] = 'center', mode='constant', value=None):
    return pad_to_shape(input, target.size(), where=where, mode=mode, value=value)

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


def pad_to_channels(input:torch.Tensor, channels:int, where:Literal['center', 'start', 'end'] = 'start', mode='constant', value=None):
    """Pads `input` to have the same shape as `target`.

    Args:
        input (torch.Tensor): Input to pad.
        target (torch.Tensor): Target that the input will match the shape of, must have same or bigger shape among all dimensions.
        where (str, optional): How to pad, if `center`, input will be in the center, etc. Defaults to 'center'.
        mode (str, optional): Padding mode from `torch.nn.functional.pad`, can be 'constant', 'reflect', 'replicate' or 'circular'. Defaults to 'constant'.
        value (_type_, optional): Constant padding value if `mode` is `constant`. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    return pad_to_shape(input, (input.shape[0], channels, *input.shape[2:]), where=where, mode=mode, value=value)

def pad_to_channels_like(input:torch.Tensor, target:torch.Tensor, where:Literal['center', 'start', 'end'] = 'center', mode='constant', value=None):
    return pad_to_channels(input, target.size(1), where=where, mode=mode, value=value)


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
