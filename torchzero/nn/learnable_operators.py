"""Hi"""
import torch
from torch import nn

__all__ = [
    'LearnableAdd',
    'LearnableMul',
    'LearnableMulAdd',
    'LearnableMulAddChannelwise',
]

class LearnableAdd(nn.Module):
    def __init__(self, init = 0., learnable=True):
        """Adds a learnable scalar to input.

        Args:
            init (_type_, optional): _description_. Defaults to 0..
            learnable (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(init, requires_grad=learnable), learnable)
    def forward(self, x):
        return x + self.bias


class LearnableMul(nn.Module):
    def __init__(self, init = 1., learnable=True):
        """Multiplies input by a learnable scalar.

        Args:
            init (_type_, optional): _description_. Defaults to 1..
            learnable (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(init, requires_grad=learnable), learnable)
    def forward(self, x):
        return x * self.weight

class LearnableMulAdd(nn.Module):
    def __init__(self, init_w = 1., init_b = 0., learnable=True):
        """Multiplies input by a learnable weight scalar and adds a learnable bias scalar.

        Args:
            init_w (_type_, optional): _description_. Defaults to 1..
            init_b (_type_, optional): _description_. Defaults to 0..
            learnable (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(init_w, requires_grad=learnable), learnable)
        self.bias = nn.Parameter(torch.tensor(init_b, requires_grad=learnable), learnable)

    def forward(self, x):
        return (x * self.weight) + self.bias

class LearnableMulAddChannelwise(nn.Module):
    def __init__(self, in_channels, ndim = 2, init_w = 1., init_b = 0., learnable=True):
        """Multiplies input by a learnable channel-wise weight vector and adds a learnable channel-wise bias vector.

        Args:
            in_channels (_type_): _description_
            ndim (int, optional): _description_. Defaults to 2.
            init_w (_type_, optional): _description_. Defaults to 1..
            init_b (_type_, optional): _description_. Defaults to 0..
            learnable (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        self.n_channels = in_channels
        self.weight = nn.Parameter(torch.full(size=(1, in_channels, *[1 for _ in range(ndim)]), fill_value=init_w, requires_grad=learnable), requires_grad=learnable)
        self.bias = nn.Parameter(torch.full(size=(1, in_channels, *[1 for _ in range(ndim)]), fill_value=init_b, requires_grad=learnable), requires_grad=learnable)
        self.learnable = learnable

    def forward(self, x):
        return (x * self.weight) + self.bias

