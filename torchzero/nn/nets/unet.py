from typing import Any
import torch
from torchzero.nn.cubes import *
from torchzero.nn.cubes.chain import _generate_channels
from ..cubes._utils import _unsupported_by_this_cube, _process_partial_seq

DownConvBlock = StraightAndResizeCube.partial(
    straight_cube=ConvCube.partial(kernel_size=2, act="relu", norm="batchnorm", dropout=0.5),
    resize_cube=MaxPoolCube,
    straight_num=2,
    channel_mode="out",
    only_straight_channels=True,
)

UpConvBlock = StraightAndResizeCube.partial(
    straight_cube=ConvCube.partial(kernel_size=2, act="relu", norm="batchnorm", dropout=0.5),
    resize_cube=ConvTransposeStrideCube.partial(act="relu", norm="batchnorm"),
    straight_num=2,
    channel_mode="in",
    only_straight_channels=False,
)

StraightConvBlock = ChainCube.partial(
    cube=ConvCube.partial(kernel_size=2, act="relu", norm="batchnorm", dropout=0.5),
    num=2,
    channel_mode="out",
)

LastConvBlock = (ConvCube.partial(kernel_size=2, act='relu', norm='batchnorm', dropout=0.5), ConvCube.partial(kernel_size=1))

class UNet(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        ndim = 2,
        levels = 4,
        first_out_channels = 64,
        hidden_channels = 192,
        last_in_channels = 64,
        middle_out_channels = None,
        first: Any = StraightConvBlock,
        down: Any = DownConvBlock,
        middle: Any = StraightConvBlock,
        up: Any = UpConvBlock,
        last: Any = LastConvBlock,
        snap_channels = 16,
        path_dropout = None,
        scale = _unsupported_by_this_cube,
    ):
        super().__init__()

        if middle_out_channels is None:
            middle_out_channels = hidden_channels

        if first is not None:
            if isinstance(first, torch.nn.Module): self.first = first
            else: self.first = _process_partial_seq(first)(in_channels=in_channels, out_channels=first_out_channels, ndim=ndim,scale=None,)
        else: self.first = None

        if last is not None:
            if isinstance(last, torch.nn.Module): self.last = last
            else: self.last = _process_partial_seq(last)(in_channels=last_in_channels, out_channels=out_channels, ndim=ndim,scale=None,)
        else: self.last = None

        if middle is not None:
            if isinstance(middle, torch.nn.Module): self.middle = middle
            else: self.middle = _process_partial_seq(middle)(in_channels=hidden_channels, out_channels=middle_out_channels, ndim=ndim, scale=None, )
        else: self.middle = None

        self.encoder = ChainCube(
            cube=_process_partial_seq(down),
            num=levels,
            in_channels=first_out_channels if self.first is not None else in_channels,
            out_channels=hidden_channels,
            ndim=ndim,
            scale=0.5,
            return_each=True,
            snap_channels=snap_channels,
        )

        self.decoder = ChainCube(
            cube=_process_partial_seq(up),
            num=levels,
            in_channels=hidden_channels if self.middle is None else middle_out_channels,
            out_channels=last_in_channels if self.last is not None else out_channels,
            ndim=ndim,
            scale=2,
            snap_channels=snap_channels,
            cat_channels=list(reversed(self.encoder.channels))[1:-1],
            cat_idxs=slice(1, None),
            cat_dropout=path_dropout,
        )

    def forward(self, x: torch.Tensor):
        if self.first is not None: x = self.first(x)
        x, each = self.encoder(x)
        if self.middle is not None: x = self.middle(x)
        x = self.decoder(x, cat_levels=list(reversed(each))[1:])
        if self.last is not None: x = self.last(x)
        return x