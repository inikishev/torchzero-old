from collections.abc import Sequence
from typing import Literal
import torch
from ...python_tools import reduce_dim


def pad(
    input: torch.Tensor,
    padding: Sequence[int],
    mode: str = "constant",
    value=None,
    where: Literal["center", "start", "end"] = "center",
) -> torch.Tensor:
    """
    Padding function that is easier to read:

    `output.shape[i]` = `input.shape[i] + padding[i]`.

    Args:
        input (torch.Tensor): input to pad.
        padding (str): How much padding to add per each dimension of `input`.
        mode (str, optional): Padding mode (https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html). Defaults to 'constant'.
        value (_type_, optional): Padding constant value. Defaults to None.
        where (str, optional): How to pad, if `center`, will pad start and end of each dimension evenly, if `start`, will pad at the start of each dimension , if `end`, will pad at the end. Defaults to 'center'.

    Returns:
        torch.Tensor: Padded `input`.
    """
    # create padding sequence for torch.nn.functional.pad
    if where == 'center':
        torch_padding = [(int(i / 2), int(i / 2)) if i % 2 == 0 else (int(i / 2), int(i / 2) + 1) for i in padding]
    elif where == 'start':
        torch_padding = [(i, 0) for i in padding]
    elif where == 'end':
        torch_padding = [(0, i) for i in padding]
    else: raise ValueError(f'Invalid where: {where}')

    # broadcasting (e.g. if padding 3×128×128 by [4, 4], it will pad by [0, 4, 4])
    while len(torch_padding) < input.ndim: torch_padding.insert(0, (0, 0))
    
    if mode == 'zeros': mode = 'constant'; value = 0

    return torch.nn.functional.pad(input, reduce_dim(reversed(torch_padding)), mode=mode, value=value)

def pad_to_shape(
    input:torch.Tensor,
    shape:Sequence[int],
    mode:str = "constant",
    value=None,
    where:Literal["center", "start", "end"] = "center",
) -> torch.Tensor:
    return pad(
        input=input,
        padding=[shape[i] - input.shape[i] for i in range(input.ndim)],
        mode=mode,
        value=value,
        where=where,
    )
    
def pad_like(input:torch.Tensor, target:torch.Tensor, where:Literal['center', 'start', 'end'] = 'center', mode='constant', value=None):
    return pad_to_shape(input, target.size(), where=where, mode=mode, value=value)


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
