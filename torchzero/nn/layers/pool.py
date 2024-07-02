import torch
from .._library.pool import AvgPoolnd, MaxPoolnd, AdaptiveAvgPoolnd, AdaptiveMaxPoolnd

class AvgMaxCatPool(torch.nn.Module):
    def __init__(self, kernel_size: int|tuple,
    stride: int|tuple | None = None,
    padding: int|tuple = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
    ndim: int = 2
):
        super().__init__()
        self.avg_pool = AvgPoolnd(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override, ndim=ndim)
        self.max_pool = MaxPoolnd(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, ndim=ndim)

    def forward(self, x):
        return torch.cat((self.avg_pool(x),  self.max_pool(x)), 1)

class AvgMaxSumPool(torch.nn.Module):
    def __init__(self, kernel_size: int|tuple,
    stride: int|tuple | None = None,
    padding: int|tuple = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
    ndim: int = 2
):
        super().__init__()
        self.avg_pool = AvgPoolnd(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override, ndim=ndim)
        self.max_pool = MaxPoolnd(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, ndim=ndim)

    def forward(self, x):
        return self.avg_pool(x) + self.max_pool(x)

class AvgMaxMulPool(torch.nn.Module):
    def __init__(self, kernel_size: int|tuple,
    stride: int|tuple | None = None,
    padding: int|tuple = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
    ndim: int = 2
):
        super().__init__()
        self.avg_pool = AvgPoolnd(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override, ndim=ndim)
        self.max_pool = MaxPoolnd(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, ndim=ndim)

    def forward(self, x):
        return self.avg_pool(x) * self.max_pool(x)


class AdaptiveAvgMaxCatPool(torch.nn.Module):
    def __init__(self, output_size: int | tuple, ndim: int = 2):
        super().__init__()
        self.max_pool = AdaptiveMaxPoolnd(output_size=output_size, ndim=ndim)
        self.avg_pool = AdaptiveAvgPoolnd(output_size=output_size, ndim=ndim)

    def forward(self, x):
        return torch.cat((self.avg_pool(x),  self.max_pool(x)), 1)

class AdaptiveAvgMaxSumPool(torch.nn.Module):
    def __init__(self, output_size: int | tuple, ndim: int = 2):
        super().__init__()
        self.max_pool = AdaptiveMaxPoolnd(output_size=output_size, ndim=ndim)
        self.avg_pool = AdaptiveAvgPoolnd(output_size=output_size, ndim=ndim)

    def forward(self, x):
        return self.avg_pool(x) + self.max_pool(x)


class AdaptiveAvgMaxMulPool(torch.nn.Module):
    def __init__(self, output_size: int | tuple, ndim: int = 2):
        super().__init__()
        self.max_pool = AdaptiveMaxPoolnd(output_size=output_size, ndim=ndim)
        self.avg_pool = AdaptiveAvgPoolnd(output_size=output_size, ndim=ndim)

    def forward(self, x):
        return self.avg_pool(x) * self.max_pool(x)
