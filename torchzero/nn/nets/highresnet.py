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
from ..cubes._utils import unsupported_by_this_cube, partial_seq, CubePartial, partial_ignore
from ..functional.pad import pad_to_channels_like

HighResBlock = ResidualCube.partial(
    cube=[ConvCube.partial(
            kernel_size = 3,
            act = "relu",
            norm = "bn",
            order = "NAC",
            padding = partial_ignore, # type:ignore
            dilation = partial_ignore, # type:ignore
        ) for _ in range(2)] + [dict(c=0)])

class HighResNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ndim: int,
        first: CubePartial = ConvCube.partial(3, 'relu', 'bn', order = 'CNA'),
        straight: CubePartial = HighResBlock,
        head: CubePartial = ConvCube.partial(1),
        channels = ([16]*3) + ([32]*3) + ([64]*3),
        padding = (['auto']*3) + ([2]*3) + ([4]*3),
        dilation = (['auto']*3) + ([2]*3) + ([4]*3),
    ):
        super().__init__()
