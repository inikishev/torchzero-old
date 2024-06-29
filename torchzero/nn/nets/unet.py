from typing import Any, Literal
import torch
from ..cubes.conv import ConvCube
from ..cubes.convtranspose import ConvTransposeStrideCube
from ..cubes.chain import ChainCube, StraightAndResizeCube
from ..cubes.chain import _generate_channels
from ..cubes._utils import _unsupported_by_this_cube, _process_partial_seq
from ..layers.pad import pad_like

class UNet(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        ndim,
        straight: Any = ConvCube.partial(3, act='relu', norm='bn', dropout=0.5, order='NADC'),
        downsample: Any = ConvCube.partial(3, resample='stride'),
        upsample: Any = ConvTransposeStrideCube.partial(),
        first: Any = ConvCube.partial(3),
        head: Any = ConvCube.partial(1),
        blocks_per_level = 2,
        middle_blocks = 3,
        encoder_channels = (32, 64, 96, 128),
        decoder_channels = (192, 128, 96, 64),
        skip: Literal['cat', 'sum'] = 'cat',
        scale = _unsupported_by_this_cube,
    ):
        """Use `norm/act/dropout -> conv` in blocks.

        UNet:
            first: conv
            downblock: [norm -> act -> dropout -> conv -> `downsample`] per `encoder_channels`
            middle: [norm -> act -> dropout -> conv] per `middle_blocks`
            upblock: [`upsample` -> norm -> act -> dropout -> conv] per `decoder_channels`
            head: conv

        Or with normal ordering:
        UNet:
            first: conv -> act -> norm -> dropout
            downlock: [conv -> act -> norm -> dropout -> `downsample`] per `encoder_channels`
            middle: [conv -> act -> norm -> dropout] per `middle_blocks`
            uplock: [`upsample` -> conv -> act -> norm -> dropout] per `decoder_channels`
            head: conv


        Args:
            in_channels (_type_): _description_
            out_channels (_type_): _description_
            ndim (_type_): _description_
            straight (Any, optional): _description_. Defaults to SingleConv.
            downsample (Any, optional): _description_. Defaults to UNetDown.
            upsample (Any, optional): _description_. Defaults to NiNBlock.
            first (Any, optional): _description_. Defaults to UNetUp.
            head (Any, optional): _description_. Defaults to LastConvBlock.
            encoder_channels (tuple, optional): _description_. Defaults to (32, 64, 128, 192).
            decoder_channels (tuple, optional): _description_. Defaults to (256, 192, 128, 64).
            scale (_type_, optional): _description_. Defaults to _unsupported_by_this_cube.

        Raises:
            ValueError: _description_
        """
        super().__init__()

        self.skip = skip

        if straight is not None: straight = _process_partial_seq(straight)
        downsample = _process_partial_seq(downsample)
        upsample = _process_partial_seq(upsample)
        if first is not None: first = _process_partial_seq(first)
        if head is not None: head = _process_partial_seq(head)

        if len(encoder_channels) != len(decoder_channels):
            raise ValueError(f"encoder_channels and decoder_channels must have the same length, {encoder_channels = }, {decoder_channels = }")

        if first is not None: self.first = first(in_channels=in_channels, out_channels=encoder_channels[0], ndim=ndim, scale=None)
        else: self.first = None

        if straight is not None:
            down_block = StraightAndResizeCube.partial(
                straight_cube=straight,
                resize_cube=downsample,
                straight_num=blocks_per_level,
                channel_mode="in",
                only_straight_channels=False,
            )
        else:
            down_block = downsample

        self.down = torch.nn.ModuleList(
            [
                down_block(
                    in_channels=encoder_channels[i],
                    out_channels=encoder_channels[i + 1],
                    ndim=ndim,
                    scale=0.5,
                )
                for i in range(len(encoder_channels) - 1)
            ]
        )

        if middle_blocks > 0:
            self.middle = ChainCube(
                in_channels=encoder_channels[-1],
                out_channels=decoder_channels[0],
                ndim = 2,
                cube=straight,
                num=middle_blocks,
                channel_mode='in'
            )
        else: 
            if encoder_channels[-1] != decoder_channels[0]: 
                raise ValueError(f"{middle_blocks = }, while ({encoder_channels[-1] = }) != ({decoder_channels[0] = })")
            self.middle = None

        if straight is not None:
            up_block = StraightAndResizeCube.partial(
                straight_cube=straight,
                resize_cube=upsample,
                straight_num=blocks_per_level,
                channel_mode="out",
                only_straight_channels=False,
                order = "RS",
            )
        else: up_block = upsample

        self.up = torch.nn.ModuleList(
            [
                up_block(
                    in_channels = encoder_channels[-i] * int(self.skip == "cat") + decoder_channels[i - 1],
                    out_channels = decoder_channels[i],
                    ndim = ndim,
                    scale = 2,
                )
                for i in range(1, len(decoder_channels))
            ]
        )

        self.head = head(
            in_channels = decoder_channels[-1] + ((encoder_channels[0] * int(self.first is not None) + in_channels) * int(self.skip == "cat")),
            out_channels = out_channels,
            ndim = ndim,
            scale = None,
        )

    def forward(self, x: torch.Tensor):
        x0 = x
        if self.first is not None: x1 = x = self.first(x)
        else: x1 = None

        each = []
        for mod in self.down:
            x = mod(x)
            each.append(x)

        if self.middle is not None: x = self.middle(x)

        for mod, y in zip(self.up, reversed(each)):
            if self.skip == 'cat': x = mod(torch.cat([x, y], dim=1))
            else: x = mod(x + pad_like(y, target=x, where='start', value=0))

        if self.skip == 'cat':
            if x1 is not None: return self.head(torch.cat([x, x1, x0], dim=1))
            return self.head(torch.cat([x, x0], dim=1))

        if x1 is not None: return self.head(x + pad_like(x1, x, 'start') + pad_like(x0, x, 'end'))
        return self.head(x + pad_like(x0, x, 'start'))