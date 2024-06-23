"""Elementwise"""
import torch
from torch import nn

__all__ = [
    'Elementwise',
    'ElementwiseAdd',
    'ElementwiseMul',
]

class Elementwise(nn.Module):
    def __init__(self, in_size, bias = True, weight_init = torch.ones, bias_init = torch.zeros):
        """Applies elementwise multiplication and addition with learnable arrays, i.e. `x * weight + bias`.

        By default initialzied with ones and zeroes and therefore computes identity function.

        `in_size`: `(C, H, W)` for 2d input.

        Args:
            in_size (_type_): _description_
            bias (bool, optional): _description_. Defaults to True.
            weight_init (_type_, optional): _description_. Defaults to torch.ones.
            bias_init (_type_, optional): _description_. Defaults to torch.zeros.
        """
        super().__init__()
        self.weight = nn.Parameter(weight_init(in_size), True)
        if bias:
            self.bias = nn.Parameter(bias_init(in_size), True)
            self.has_bias = True
        else:
            self.has_bias = False

    def forward(self, x):
        if self.has_bias: return x * self.weight + self.bias
        else: return x * self.weight

class ElementwiseAdd(nn.Module):
    def __init__(self, in_size, init = torch.zeros):
        """Applies elementwise addition with a learnable array, i.e. `x + bias`.

        `in_size`: `(C, H, W)` for 2d input.

        Args:
            in_size (_type_): _description_
            init (_type_, optional): _description_. Defaults to torch.zeros.
        """
        super().__init__()
        self.bias = nn.Parameter(init(in_size), True)

    def forward(self, x):
        return x + self.bias


class ElementwiseMul(nn.Module):
    def __init__(self, in_size, init = torch.zeros):
        """Applies elementwise multiplication with a learnable array, i.e. `x + weight`.

        `in_size`: `(C, H, W)` for 2d input.

        Args:
            in_size (_type_): _description_
            init (_type_, optional): _description_. Defaults to torch.zeros.
        """
        super().__init__()
        self.weight = nn.Parameter(init(in_size), True)

    def forward(self, x):
        return x + self.weight

