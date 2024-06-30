from typing import Any, Literal
from collections.abc import Sequence
import torch
from ..cubes.conv import ConvCube
from ..cubes.convtranspose import ConvTransposeStrideCube
from ..cubes.maxpool import MaxPoolCube
from ..cubes.chain import ChainCube, StraightAndResizeCube
from ..cubes.chain import _generate_channels
from ..cubes._utils import _unsupported_by_this_cube, _process_partial_seq, PartialIgnore
from ..layers.pad import pad_to_channels_like

class UNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ndim: int,
        straight: Any = ConvCube.partial(3, act="relu", norm="bn"),
        downsample: Any = MaxPoolCube.partial(),
        upsample: Any = ConvTransposeStrideCube.partial(),
        head: Any = ConvCube.partial(1),
        encoder_blocks: int | Sequence[int] = 2,
        decoder_blocks: int | Sequence[int] = 2,
        encoder_channels: Sequence[int] = (64, 128, 256, 512, 1024),
        decoder_channels: Sequence[int]  = (512, 256, 128, 64),
        skip: Literal["cat", "sum"] = "cat",
        decoder_channels_module: Literal["upsample", "straight"] = 'upsample',
        straight1_override: Any = None,
        first_override: Any = None,
        scale = _unsupported_by_this_cube,
    ):
        """U-Net (https://arxiv.org/pdf/1505.04597). Default arguments should create the same model as described in the paper.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            ndim (int): Number of dimensions.
            straight (Any, optional): Straight cube, may change number of channels. Defaults to ConvCube.partial(3, act="relu", norm="bn").
            downsample (Any, optional): Downsample cube. Defaults to MaxPoolCube.partial().
            upsample (Any, optional): Upsample cube. Defaults to ConvTransposeStrideCube.partial().
            head (Any, optional): Head cube. Defaults to ConvCube.partial(1).
            encoder_blocks (int | Sequence[int], optional): Number of blocks per encoder level. Defaults to 2.
            decoder_blocks (int | Sequence[int], optional): Number of blocks per decoder level. Defaults to 2.
            encoder_channels (Sequence[int], optional): Number of output channels of each encoder level. Defaults to (64, 128, 256, 512, 1024).
            decoder_channels (Sequence[int], optional): Number of output channels of each decoder level. Defaults to (512, 256, 128, 64).
            skip: "cat" for channel concatenation, "sum" for summation like in SegResNet. Defaults to "cat".
            decoder_channels_module: Which module changes number of channels.
            In original U-Net upsampling is done via transposed conv which also reduces number of channels.
            If you use something like billinear upsampling, that can't change number of channels, so you can set this to `straight`.
            Defaults to 'upsample'.
            straight1_override (Any, optional): Override for first straight cube in each encoder level. Defaults to None.
            first_override (Any, optional): Override for the first cube of the network. Defaults to None.
            scale (_type_, optional): Only use if you are using this U-Net as a cube. Defaults to _unsupported_by_this_cube.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """
        super().__init__()

        # ------------------------------- process args ------------------------------- #
        self.skip = skip
        if straight is not None: straight = _process_partial_seq(straight)
        downsample = _process_partial_seq(downsample)
        upsample = _process_partial_seq(upsample)
        if first_override is not None: first_override = _process_partial_seq(first_override)
        if head is not None: head = _process_partial_seq(head)
        if straight1_override is None: straight1_override = straight
        if first_override is None: first_override = straight

        if isinstance(encoder_blocks, int): encoder_blocks = [encoder_blocks] * len(encoder_channels)
        if isinstance(decoder_blocks, int): decoder_blocks = [decoder_blocks] * len(decoder_channels)

        # ------------------------------- validate args ------------------------------ #
        if len(encoder_channels) - 1 != len(decoder_channels):
            raise ValueError(f"encoder_channels must be one longer than decoder_channels, {encoder_channels = }, {decoder_channels = }")
        if len(encoder_blocks) != len(encoder_channels):
            raise ValueError(f"encoder_blocks must have the same length as encoder_channels, {encoder_channels = }, {encoder_blocks = }")
        if len(decoder_blocks) != len(decoder_channels):
            raise ValueError(f"decoder_channels must have the same length as decoder_blocks, {decoder_channels = }, {decoder_blocks = }")

        # ----------------------------- construct encoder ---------------------------- #
        # last block doesn't have a downsample and doesnt change number of channels
        encoder_channels = [in_channels] + list(encoder_channels) + [encoder_channels[-1]]

        self.encoder = torch.nn.ModuleList()
        for leveli in range(len(encoder_blocks)):
            block = torch.nn.ModuleDict()

            # create straight block
            block['straight'] = torch.nn.Sequential()
            for i in range(encoder_blocks[leveli]):
                # first block changes number of channels, and we set new number to `downsample_in_channels`
                if i == 0:
                    # on first level, if there is `first` block, use it instead
                    if leveli == 0:
                        block['straight'].append(first_override(in_channels=encoder_channels[leveli], out_channels=encoder_channels[leveli + 1], ndim=ndim, scale=None))
                    else:
                        block['straight'].append(straight1_override(in_channels=encoder_channels[leveli], out_channels=encoder_channels[leveli + 1], ndim=ndim, scale=None))
                # next blocks don't change number of channels
                else:
                    block['straight'].append(straight(in_channels=encoder_channels[leveli + 1], out_channels=None, ndim=ndim, scale=None))

            # we don't downsample on last block
            if leveli == len(encoder_blocks) - 1: block['downsample'] = torch.nn.Identity()
            # if not last, create downsample block
            else: block['downsample'] = downsample(in_channels=encoder_channels[leveli + 1], out_channels=None, ndim=ndim, scale=1/2)

            self.encoder.append(block)

        # restore original encoder_channels
        encoder_channels = encoder_channels[1:-1]

        # ----------------------------- construct decoder ---------------------------- #
        # first decoder block takes in signal with last encoder channels
        decoder_channels = [encoder_channels[-1]] + list(decoder_channels)

        self.decoder = torch.nn.ModuleList()
        for leveli in range(len(decoder_blocks)):
            block = torch.nn.ModuleDict()

            # determine whether upsample sets channels
            if decoder_channels_module == 'straight':
                upsample_out = None
                straight_in = decoder_channels[leveli]
            else:
                upsample_out = decoder_channels[leveli + 1]
                straight_in = upsample_out
            block['upsample'] = upsample(in_channels=decoder_channels[leveli], out_channels=upsample_out, ndim=ndim, scale=2)

            in_channels_after_skip = straight_in + encoder_channels[- leveli - 2] * (int(skip == 'cat'))
            straight_out = decoder_channels[leveli + 1]

            # create straight block
            block['straight'] = torch.nn.Sequential()
            for i in range(decoder_blocks[leveli]):
                # use straight block as head if straight is None
                if i == decoder_blocks[leveli] - 1 and leveli == len(decoder_blocks) - 1 and head is None:
                    in_ch = in_channels_after_skip if i == 0 else straight_in
                    block['straight'].append(straight1_override(in_channels=in_ch, out_channels=out_channels, ndim=ndim, scale=None))

                # first block changes number of channels, and we set new number to `downsample_in_channels`
                elif i == 0:
                    block['straight'].append(straight1_override(in_channels=in_channels_after_skip, out_channels=straight_out, ndim=ndim, scale=None))

                # next blocks don't change number of channels
                else:
                    block['straight'].append(straight(in_channels=straight_out, out_channels=None, ndim=ndim, scale=None))

            self.decoder.append(block)

        # head creates appropriate number of out channels
        if head is not None: self.head = head(in_channels=decoder_channels[-1], out_channels=out_channels, ndim=ndim, scale=None)
        else: self.head = torch.nn.Identity()


    def forward(self, x:torch.Tensor):
        each = []

        for mod in self.encoder:
            x = mod['straight'](x) # type:ignore
            each.append(x)
            x = mod['downsample'](x) # type:ignore

        each.pop()

        for mod in self.decoder:
            x = mod['upsample'](x) # type:ignore
            if self.skip == 'cat': x = torch.cat([x, each.pop()], dim=1)
            else: x = x + pad_to_channels_like(each.pop(), x, 'start')
            x = mod['straight'](x) # type:ignore

        return self.head(x)
