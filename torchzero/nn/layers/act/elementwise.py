import functools
from typing import Optional
from torch import nn
import torch.nn.functional as F
import torch

__all__ = [
    "ElementwiseReLU",
    "ElementwiseLeakyReLU",
    "ElementwiseResidualReLU",
    "ElementwiseLeakyResidualReLU",
]

class ElementwiseReLU(nn.Module):
    """Random ReLU: uses a random array as separate ReLU for each individual pixel with random thresholds.

    `in_size`: `(C, H, W)` for 2d inputs.

    Array can be learnable if `learnable` is `True`."""
    def __init__(self, in_size, learnable = True, init = functools.partial(torch.nn.init.uniform_, a=-1, b=1)):
        super().__init__()
        self.thresholds = torch.nn.Parameter(torch.zeros(in_size), requires_grad = learnable)
        init(self.thresholds)

    def forward(self, x):
        return torch.maximum(x, self.thresholds)

class ElementwiseLeakyReLU(nn.Module):
    """Random leaky ReLU: uses a random array as separate ReLU for each individual pixel with random thresholds.

    `in_size`: `(C, H, W)` for 2d inputs."""
    def __init__(self, in_size, leak=0.1, learnable = True, init = functools.partial(torch.nn.init.uniform_, a=-1, b=1), ):
        super().__init__()
        self.thresholds = torch.nn.Parameter(torch.zeros(in_size), requires_grad = learnable)
        init(self.thresholds)
        self.leak = leak

    def forward(self, x):
        return torch.maximum(x, self.thresholds) + torch.minimum(x, self.thresholds) * self.leak


class ElementwiseResidualReLU(nn.Module):
    """Random residual leaky ReLU: uses a random array as separate ReLU for each individual pixel with random thresholds.

    Computes `ElementwiseLeakyReLU(x) + x * residual`

    `in_size`: `(C, H, W)` for 2d inputs.

    Array can be learnable if `learnable` is `True`. Residual connection means learnable array can be initialized with zeroes."""
    def __init__(self, in_size, residual = 0.5, learnable = True, init = torch.nn.init.zeros_ ):
        super().__init__()
        self.thresholds = torch.nn.Parameter(torch.zeros(in_size), requires_grad = learnable)
        init(self.thresholds)
        self.residual = residual

    def forward(self, x):
        return torch.maximum(x, self.thresholds) + x * self.residual

class ElementwiseLeakyResidualReLU(nn.Module):
    """Random residual leaky ReLU: uses a random array as separate ReLU for each individual pixel with random thresholds.

    Computes `ElementwiseLeakyReLU(x) + x * residual`

    `in_size`: `(C, H, W)` for 2d inputs.

    Array can be learnable if `learnable` is `True`. Residual connection means learnable array can be initialized with zeroes."""
    def __init__(self, in_size, leak=0.1, residual = 0.5, learnable = True, init = torch.nn.init.zeros_ ):
        super().__init__()
        self.thresholds = torch.nn.Parameter(torch.zeros(in_size), requires_grad = learnable)
        init(self.thresholds)
        self.residual = residual
        self.leak = leak

    def forward(self, x):
        return torch.maximum(x, self.thresholds) + torch.minimum(x, self.thresholds) * self.leak + x * self.residual

if __name__ == "__main__":
    EwReLU = ElementwiseLeakyReLU((3, 28, 28), 0.1, True)
    batch = torch.randn(16, 3, 28, 28)
    print(EwReLU(batch).shape)
