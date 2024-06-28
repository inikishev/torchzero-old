import torch
from torch import nn
import torchvision.transforms.v2

__all__ = [
    'Affine',
]

class Affine(nn.Module):
    def __init__(self):
        """Untested, probably doesn't work."""
        super().__init__()
        self.angle = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
        self.translate = nn.Parameter(torch.tensor([0,0], dtype=torch.float32), requires_grad=True)
        self.scale = nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=True)
        self.shear = nn.Parameter(torch.tensor([0,0], dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        return torchvision.transforms.v2.functional.affine(x, 
                    angle=self.angle, # type:ignore
                    translate=self.translate,  # type:ignore
                    scale=self.scale,  # type:ignore
                    shear=self.shear) # type:ignore