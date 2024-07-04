"""Basic blocks"""
import random
from typing import Optional, Literal
from collections.abc import Sequence, Callable
import torch

from .._library.convolution import Convnd
from .conv import ConvBlock, _get_samesize_padding_int
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

__all__ = [
    "NCAConv",
]

class NCAConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int | tuple = 3,
        stride: int | tuple = 1,
        padding: int | float | tuple | str = 'auto',
        dilation: int | tuple = 1,
        groups: int = 1,
        bias: bool = True,
        ndim = 2,
        init_conv: torch.nn.Module | Callable = Convnd,
        init_groups: int | Literal['in_channels'] = 'in_channels',
        init_filters: int = 4,
        brain_conv: torch.nn.Module | Callable = Convnd,
        brain_kernel_size: int | tuple = 1,
        brain_layers = 2,
        brain_hidden_channels: int | Literal['in_channels', 'out_channels', 'init_channels'] | Sequence[int | Literal['in_channels', 'out_channels', 'init_channels']] = 'out_channels',
        brain_act = 'relu',
        mask_drop_p:float = 0.5,
        nsteps:int | tuple[int, int] = (4, 8),
        dtype = None,
        device = None,
        out_channels = None,
    ):
        """Neural Cellular Automata convolution.

        Args:
            in_channels (Optional[int], optional): _description_. Defaults to None.
            out_channels (Optional[int], optional): _description_. Defaults to None.
            kernel_size (int | tuple, optional): _description_. Defaults to 3.
            stride (int | tuple, optional): _description_. Defaults to 1.
            padding (int | tuple | str, optional): _description_. Defaults to 0.
            dilation (int | tuple, optional): _description_. Defaults to 1.
            groups (int, optional): _description_. Defaults to 1.
            bias (bool, optional): _description_. Defaults to True.
            padding_mode (str, optional): _description_. Defaults to 'zeros'.
            device (_type_, optional): _description_. Defaults to None.
            dtype (_type_, optional): _description_. Defaults to None.
            conv (torch.nn.Module | Callable | Literal[&#39;torch&#39;], optional): _description_. Defaults to 'torch'.
            ndim (int, optional): _description_. Defaults to 2.
        """
        super().__init__()
        if out_channels is not None and out_channels != in_channels:
            raise ValueError(f"NCAConv doesn't support out_channels different from in channels, {in_channels = }, {out_channels = }")
        if out_channels is None: out_channels = in_channels

        if isinstance(brain_hidden_channels, (int, str)) and brain_layers > 1:
            brain_hidden_channels = [brain_hidden_channels for _ in range(brain_layers - 1)]

        brain_hidden_channels = [(in_channels if i == 'in_channels' else i) for i in brain_hidden_channels] # type:ignore
        brain_hidden_channels = [(out_channels if i == 'out_channels' else i) for i in brain_hidden_channels] # type:ignore
        brain_hidden_channels = [(in_channels * init_filters if i == 'init_channels' else i) for i in brain_hidden_channels] # type:ignore
        brain_channels = [in_channels * init_filters] + brain_hidden_channels + [out_channels]

        if padding == 'auto': padding = _get_samesize_padding_int(kernel_size)

        self.init_conv = init_conv(
            in_channels = in_channels,
            out_channels = in_channels * init_filters,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = init_groups if isinstance(init_groups, int) else in_channels,
            bias = bias,
            device = device,
            dtype = dtype,
            ndim = ndim,
        )
        self.brain = torch.nn.Sequential()
        for i in range(brain_layers):
            self.brain.append(ConvBlock(
                in_channels = brain_channels[i],
                out_channels = brain_channels[i + 1],
                kernel_size = brain_kernel_size,
                groups = groups,
                bias = i != brain_layers - 1,
                device = device,
                ndim = ndim,
                act = brain_act if i != brain_layers - 1 else None, # type:ignore
                conv=brain_conv
            ))

        self.mask_keep_p = 1 - mask_drop_p
        self.nsteps = nsteps

    def one_step(self, x:torch.Tensor) -> torch.Tensor:
        #y = self.brain(self.init_conv(x))
        y = self.init_conv(x)
        y = self.brain(y)
        mask_shape = list(x.size())
        mask_shape[1] = 1
        mask = (torch.rand(size=mask_shape, device=x.device, dtype=x.dtype) + self.mask_keep_p).floor()
        return x + y * mask

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if isinstance(self.nsteps, tuple): nsteps = random.randint(*self.nsteps)
        else: nsteps = self.nsteps
        for _ in range(nsteps):
            x = self.one_step(x)
        return x
