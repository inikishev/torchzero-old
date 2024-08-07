from collections.abc import Sequence
from typing import Literal
import torch

def crop_dims(input:torch.Tensor, crop:Sequence[int], where: Literal["center", "start", "end"] = "center",):
    """Crop `input` using `crop`:

    `output.shape[i]` = `input.shape[i] - crop[i]`.

    """
    print(crop)
    if where == 'center':
        slices = [(int(i / 2), -int(i / 2)) if i % 2 == 0 else (int(i / 2), -int(i / 2) - 1) for i in crop]
    elif where == 'start':
        slices = [(None, -i) for i in crop]
    elif where == 'end':
        slices = [(i, None) for i in crop]

    slices = [slice(i if i!=0 else None, j if j != 0 else None) for i, j in slices]

    return input[(..., *slices)]

def crop_to_shape(input:torch.Tensor, shape:Sequence[int], where: Literal["center", "start", "end"] = "center",):
    """Crop `input` to `shape`."""
    return crop_dims(input, [i - j for i, j in zip(input.shape, shape)], where=where)

def crop_like(input:torch.Tensor, target:torch.Tensor, where: Literal["center", "start", "end"] = "center",):
    """Crop `input` to `target.shape`."""
    return crop_to_shape(input, target.shape, where=where)

def spatial_crop(x:torch.Tensor, crop:int = 1):
    """Crops spatial dim sizes in a BC* tensor by `num`.
    For example, if `num = 1`, (16, 3, 129, 129) -> (16, 3, 128, 128).
    This crops at the end. Useful to crop padding which can only add even size."""
    slices = [slice(None, -crop) for _ in range(x.ndim - 2)]
    return x[:,:,*slices]
