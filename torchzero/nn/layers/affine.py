from typing import Literal
import torch
from torch import nn

class Affine(nn.Module):
    def __init__(self, scalew=1., scaleh=1., transX=0., transY=0., transZ = 0., ndim = 2, mode='bilinear', padding_mode='zeros',align_corners=True):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

        #if ndim == 1: self.theta = nn.Parameter(torch.tensor([[scalew, transX]]), requires_grad=True) # not supported
        if ndim == 2:
            self.theta = nn.Parameter(
                torch.tensor([[scalew, 0., transX], [0., scaleh, transY]]),
                requires_grad=True)
            self.expand_values = [2, 3]
        elif ndim == 3:
            self.theta = nn.Parameter(
                torch.tensor([[scalew, 0., 0., transX], [0., scaleh, 0., transY], [0., 0., 1., transZ]]),
                requires_grad=True)
            self.expand_values = [3, 4]
        else: raise ValueError(f"ndim must be 2 or 3, got {ndim}")

    def forward(self, x):
        grid = torch.nn.functional.affine_grid(self.theta.expand(x.size(0), *self.expand_values), x.size(), align_corners=self.align_corners)
        return torch.nn.functional.grid_sample(x, grid, mode = self.mode, padding_mode=self.padding_mode, align_corners=self.align_corners)

class AffineNet(nn.Module):
    def __init__(self, out_channels, ndim=2, mode='bilinear', padding_mode='zeros', align_corners=True, 
                 agg : Literal['mean', 'min', 'max', 'sum', 'prod', 'cat', None]='mean'
                 ):
        super().__init__()
        self.ndim = ndim
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.out_channels = out_channels
        self.agg = agg

        if ndim == 2:
            self.theta = nn.Parameter(
                torch.randn((out_channels, 1, 2, 3)),
                requires_grad=True)
            self.expand_values = [2, 3]
        elif ndim == 3:
            self.theta = nn.Parameter(
                torch.randn((out_channels, 1, 3, 4)),
                requires_grad=True)
            self.expand_values = [3, 4]
        else: raise ValueError(f"ndim must be 2 or 3, got {ndim}")

    def forward(self, x:torch.Tensor):

        size = x.size()
        res = [
            torch.nn.functional.grid_sample(x,
                torch.nn.functional.affine_grid(self.theta[ch].expand(x.size(0), *self.expand_values), size, align_corners=self.align_corners),  #type:ignore
                mode = self.mode, padding_mode=self.padding_mode, align_corners=self.align_corners)
            for ch in range(self.out_channels)
        ]

        if self.agg == 'mean':
            return torch.stack(res, 1).mean(dim=2)
        elif self.agg == 'sum':
            return torch.stack(res, 1).sum(dim=2)
        elif self.agg == 'max':
            return torch.stack(res, 1).max(dim=2)
        elif self.agg == 'min':
            return torch.stack(res, 1).min(dim=2)
        elif self.agg == 'prod':
            return torch.stack(res, 1).prod(dim=2)
        elif self.agg == 'cat' or self.agg is None:
            return torch.cat(res, 1)