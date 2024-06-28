from collections.abc import Sequence
from typing import Optional, Any
import torch
from torch import nn

def create_upsample(module, num_channels:Optional[int], ndim:Optional[int], spatial_size:Optional[Sequence[int]], scale:Any = 2):
    if callable(module) or module is None: return module
    if module in ('nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'): return nn.Upsample(scale_factor = scale, mode = module)
    if isinstance(module, (int, float)): return nn.Upsample(scale_factor = module, mode='bilinear')
    else: raise ValueError(f'Unknown upsample module: {module}')

