from typing import Any, Literal, Optional
from collections.abc import Sequence, Mapping
import torch
from ..cubes.conv import ConvCube
from ..cubes.residual import ResidualCube
from ..cubes.convtranspose import ConvTranspose2StrideCube, ConvTranspose3StrideCube
from ..cubes.maxpool import MaxPoolCube
from ..cubes.chain import ChainCube, StraightAndResizeCube, SequenceCube, _generate_channels
from ..cubes.skip import SkipCube, SkipLiteral
from ..cubes.identity import IdentityCube
from ..cubes._utils import unsupported_by_this_cube, partial_seq, CubePartial
from ..layers.pad import pad_to_channels_like



class BasicUNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ndim: int,
        first: Optional[CubePartial] = ConvCube.partial(3, 'relu', 'bn'),
        down: CubePartial = ConvCube.partial(3, 'relu', 'bn'),
        up: CubePartial = ConvTranspose2StrideCube.partial('relu', 'bn'),
        head: Optional[CubePartial] = ConvCube.partial(1),
        encoder_channels=(32, 64, 128, 256),
        decoder_channels=(192, 96, 48),
        skip_type: SkipLiteral = "cat",
        skip_cube: Optional[CubePartial] = None,
        skip_dropout: Optional[float] = None,
        skip_orig_dropout: Optional[float] = None,
        skip_swap_p: Optional[float] = None,
        encoder_overrides: Optional[Mapping[int | slice, CubePartial]] = None,
        decoder_overrides: Optional[Mapping[int | slice, CubePartial]] = None,
        level1_skip: bool = True,
        level1_skip_input: bool = False,
        scale = unsupported_by_this_cube,
    ):
        super().__init__()
        # ------------------------------- process args ------------------------------- #
        up = partial_seq(up); down = partial_seq(down)
        if first is not None: first = partial_seq(first)
        if head is not None: head = partial_seq(head)
        if skip_cube is not None: skip_cube = partial_seq(skip_cube)
        if first is None and encoder_channels[0] != in_channels: raise ValueError(f"first cube must be provided if encoder_channels[0] != in_channels, {encoder_channels[0] = }, {in_channels = }")
        if head is None and decoder_channels[-1] != out_channels: raise ValueError(f"head cube must be provided if decoder_channels[-1] != out_channels, {decoder_channels[-1] = }, {out_channels = }")
        if encoder_overrides is None: encoder_overrides = {}
        if decoder_overrides is None: decoder_overrides = {}
        if len(encoder_channels) - 1 != len(decoder_channels): raise ValueError(f"encoder and decoder channel counts must match, {encoder_channels = }, {decoder_channels = }")

        encoder_cubes = [down for _ in range(len(encoder_channels) - 1)]
        for k,v in encoder_overrides.items():
            if isinstance(k, slice):
                for i in range(*k.indices(len(encoder_cubes))): encoder_cubes[i] = partial_seq(v)
            else: encoder_cubes[k] = partial_seq(v)

        decoder_cubes = [up for _ in range(len(decoder_channels))]
        for k,v in decoder_overrides.items():
            if isinstance(k, slice):
                for i in range(*k.indices(len(decoder_cubes))): decoder_cubes[i] = partial_seq(v)
            else: decoder_cubes[k] = partial_seq(v)

        self.level1_skip_input = level1_skip_input

        # ------------------------------ create modules ------------------------------ #
        if first is not None: self.first = first(in_channels = in_channels, out_channels = encoder_channels[0], ndim=ndim, scale=None) # type:ignore
        else: self.first = torch.nn.Identity()
        self.encoder = torch.nn.ModuleList([cube(in_channels=encoder_channels[i], out_channels=encoder_channels[i+1], ndim=ndim, scale = 0.5) for i, cube in enumerate(encoder_cubes)])

        # first cube doesn't have a skip connection
        self.decoder = torch.nn.ModuleList([decoder_cubes[0](in_channels=encoder_channels[-1], out_channels=decoder_channels[0], scale=2, ndim=ndim)])
        for i, cube in enumerate(decoder_cubes[1:]):
            self.decoder.append(SkipCube(
                cube = cube,
                in_channels = decoder_channels[i],
                out_channels = decoder_channels[i + 1],
                skip_channels = encoder_channels[-i - 2],
                ndim = ndim,
                scale = 2,
                skip_mode = skip_type,
                skip_cube = skip_cube,
                skip_dropout = skip_dropout,
                orig_dropout = skip_orig_dropout,
                swap_p = skip_swap_p,
            ))

        if head is not None:
            self.head = SkipCube(
                cube = head,
                in_channels=decoder_channels[-1],
                out_channels=out_channels,
                skip_channels=encoder_channels[0] + (in_channels * int(level1_skip_input)),
                ndim=ndim,
                scale=None,
                skip_mode = skip_type if level1_skip else 'none',
                skip_cube = skip_cube,
                skip_dropout = skip_dropout,
                orig_dropout = skip_orig_dropout,
                swap_p = skip_swap_p,
                )
        else: self.head = torch.nn.Identity()

    def forward(self, x:torch.Tensor):
        skips = [self.first(x)]
        for m in self.encoder: skips.append(m(skips[-1]))
        y = self.decoder[0](skips.pop())
        for m in self.decoder[1:]: y = m(y, skips.pop()) # type:ignore
        if self.level1_skip_input: return self.head(y, torch.cat((skips[0], x), dim = 1))
        return self.head(y, skips[0])


class UNet(BasicUNet):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ndim: int,
        straight: CubePartial = ConvCube.partial(3, act="relu", norm="bn"),
        downsample: CubePartial = MaxPoolCube.partial(),
        upsample: CubePartial = ConvTranspose2StrideCube.partial(),
        head: Optional[CubePartial] = ConvCube.partial(1),
        encoder_blocks: int | Sequence[int] = 2,
        decoder_blocks: int | Sequence[int] = 2,
        encoder_channels: Sequence[int] = (64, 128, 256, 512, 1024),
        decoder_channels: Sequence[int]  = (512, 256, 128, 64),
        skip: SkipLiteral = "cat",
        skip_cube: Optional[CubePartial] = None,
        skip_dropout = None,
        skip_orig_dropout = None,
        skip_swap_p = None,
        encoder_channels_module: Literal["downsample", "straight"] = 'straight',
        decoder_channels_module: Literal["upsample", "straight"] = 'upsample',
        decoder_straight1_override: Optional[CubePartial] = None,
        first_override: Optional[CubePartial] = None,
        scale = unsupported_by_this_cube,
    ):
        # ------------------------------- process args ------------------------------- #
        if isinstance(encoder_blocks, int): encoder_blocks = [encoder_blocks for _ in  range(len(encoder_channels))]
        if isinstance(decoder_blocks, int): decoder_blocks = [decoder_blocks for _ in  range(len(decoder_channels))]
        straight = partial_seq(straight); downsample = partial_seq(downsample); upsample = partial_seq(upsample)
        if decoder_channels_module == 'straight': raise NotImplementedError("straight not implemented for decoder yet")
        if len(encoder_blocks) != len(encoder_channels): raise ValueError(f"encoder_blocks must have same length as encoder_channels - 1, {encoder_blocks = }, {encoder_channels = }")
        if len(decoder_blocks) != len(decoder_channels): raise ValueError(f"decoder_blocks must have same length as decoder_channels, {decoder_blocks = }, {decoder_channels = }")

        first_blocks, encoder_blocks = encoder_blocks[0], encoder_blocks[1:]
        # ------------------------------- create blocks ------------------------------ #

        encoder_cubes: Mapping[int | slice, CubePartial] = {i: StraightAndResizeCube.partial(
            straight_cube = straight,
            resize_cube = downsample,
            straight_num = num,
            channel_mode='out',
            order = 'RS',
            only_straight_channels = (encoder_channels_module == 'straight'),
            ) for i, num in enumerate(encoder_blocks)}

        decoder_cubes: dict[int | slice, CubePartial] = {}
        decoder_cubes[0] = upsample
        for i, num in enumerate(decoder_blocks[:-1], 1):
            cubes = [*[straight for _ in range(num)], upsample]
            if len(cubes) > 1 and decoder_straight1_override is not None: cubes[0] = partial_seq(decoder_straight1_override)
            decoder_cubes[i] = SequenceCube.partial(
                cubes = cubes,
                mid_channels = decoder_channels[i - 1] if decoder_channels_module == 'upsample' else (decoder_channels[i - 2] if i != 1 else encoder_channels[-1]),
                scale_idx=-1,
                )

        if head is None: head = straight
        else:
            cubes = [*[straight for _ in range(decoder_blocks[-1])], head]
            if len(cubes) > 1 and decoder_straight1_override is not None: cubes[0] = partial_seq(decoder_straight1_override)
            head = SequenceCube.partial(
                cubes = cubes,
                mid_channels = decoder_channels[-1]
                )

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            ndim=ndim,
            first = [straight for _ in range(first_blocks)]
                if first_override is None
                else [partial_seq(first_override) for _ in range(first_blocks)],
            down=IdentityCube,
            up=IdentityCube,
            head=head,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            skip_type=skip,
            skip_cube=skip_cube,
            skip_dropout=skip_dropout,
            skip_orig_dropout=skip_orig_dropout,
            skip_swap_p=skip_swap_p,
            encoder_overrides=encoder_cubes,
            decoder_overrides=decoder_cubes,
            scale=scale,
        )


SegResBlock = ResidualCube.partial(
    ChainCube.partial(
        cube = ConvCube.partial(
            kernel_size = 3,
            act = "relu",
            norm = "bn",
            order="NADC",
            bias=False
        ),
        num = 2,
        channel_mode = 'max',
    )
)

class SegResNet(UNet):
    def __init__(self,
        in_channels = 12,
        out_channels = 5,
        ndim = 2,
        straight = SegResBlock,
        downsample = ConvCube.partial(3, resample='stride', bias=False),
        upsample = ConvTranspose3StrideCube.partial(bias = False),
        head = ConvCube.partial(1),
        first = ConvCube.partial(3, bias=False),
        skip: SkipLiteral = 'sum',
        skip_cube: CubePartial | None = None,
        skip_dropout: Optional[float] = None,
        skip_orig_dropout: Optional[float] | None = None,
        skip_swap_p: Optional[float] | None = None,
        encoder_channels_module: Literal['downsample', 'straight',] = 'downsample',
        decoder_channels_module: Literal['upsample', 'straight',] = 'upsample',
        encoder_channels = (32, 64, 128, 256),
        decoder_channels = (128, 64, 32),
        encoder_blocks = (1, 2, 2, 4),
        decoder_blocks = 1,
        scale = unsupported_by_this_cube,
        ):
        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            ndim = ndim,
            straight = straight,
            downsample = downsample,
            upsample = upsample,
            head = head,
            first_override = [first, SegResBlock, dict(c=0)],
            skip = skip,
            skip_cube = skip_cube,
            skip_dropout = skip_dropout,
            skip_orig_dropout = skip_orig_dropout,
            skip_swap_p = skip_swap_p,
            encoder_channels_module = encoder_channels_module,
            decoder_channels_module = decoder_channels_module,
            encoder_channels = encoder_channels,
            decoder_channels = decoder_channels,
            encoder_blocks = encoder_blocks,
            decoder_blocks = decoder_blocks,
            scale = scale,
        )
