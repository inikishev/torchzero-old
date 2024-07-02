"""Basic blocks"""
from typing import Optional, Literal
from collections.abc import Sequence, Callable
import torch
import math

# conv2d args:
# in_channels: int, 
# out_channels: int, 
# kernel_size: int | tuple, 
# stride: int | tuple = 1,
# padding: int | tuple | str = 0,
# dilation: int | tuple = 1, 
# groups: int = 1, 
# bias: bool = True, 
# padding_mode: str = 'zeros', 
# device = None, 
# dtype = None

class AgentConv(torch.nn.Module):
    """AgentConv module."""
    def __init__(
        self,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        kernel_size: int | tuple = 3,
        stride: int | tuple = 1,
        padding: int | tuple | str = 0,
        dilation: int | tuple = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device = None,
        dtype = None,
        conv: torch.nn.Module | Callable | Literal['torch'] = 'torch',
        ndim = 2,
    ):
        super().__init__()
        ...
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        ...