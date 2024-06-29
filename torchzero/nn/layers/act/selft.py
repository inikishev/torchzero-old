"""Self-thresholding activation functions."""
from typing import Optional
import torch

def half_minmax(x:torch.Tensor, residual: Optional[float | torch.Tensor]=None):
    channels = x.size(1)
    if channels == 1: return torch.nn.functional.relu(x, True)
    if channels % 2 == 1: raise ValueError(f"`half_minmax` got an odd number of channels: {channels = }")
    half = int(channels / 2)
    half1, half2 = x[:,:half], x[:,half:]

    if residual is None: return torch.cat((torch.minimum(half1, half2), torch.maximum(half1, half2)), dim=1)
    else: return torch.cat((torch.minimum(half1, half2), torch.maximum(half1, half2)), dim=1) + x * residual

class HalfMinMax(torch.nn.Module):
    def __init__(self, residual:Optional[float | torch.Tensor]=None):
        """Splits input into two halves by channels.

        Then returns (in pseudocode):

        `channel_concat([elementwise_min(half1, half2), elementwise_max(half1, half2)]) + (input * residual)`

        Args:
            residual (Optional[float  |  torch.Tensor]): Let some of the input go through. This is sort of like leak. Defaults to None.
        """
        super().__init__()
        self.residual = residual
    def forward(self, x:torch.Tensor):
        return half_minmax(x, self.residual)

def half_summax(x:torch.Tensor, residual: Optional[float | torch.Tensor]=None):
    channels = x.size(1)
    if channels == 1: return torch.nn.functional.relu(x, True)
    if channels % 2 == 1: raise ValueError(f"`half_summax` got an odd number of channels: {channels = }")
    half = int(channels / 2)
    half1, half2 = x[:,:half], x[:,half:]
    sum = half1 + half2
    if residual is None: return torch.cat((torch.maximum(half1, sum), torch.maximum(half2, sum)), dim=1)
    return torch.cat((torch.maximum(half1, sum), torch.maximum(half2, sum)), dim=1) + x * residual

class HalfSumMax(torch.nn.Module):
    def __init__(self, residual:Optional[float | torch.Tensor]=None):
        """Splits input into two halves by channels, calculates their sum, and returns element-wise maximums of each half with the sum.

        Then returns (in pseudocode):

        `channel_concat([elementwise_max(half1, half1 + half2), elementwise_max(half2, half1 + half2)]) + (input * residual)`

        Args:
            residual (Optional[float  |  torch.Tensor]): Let some of the input go through. This is sort of like leak. Defaults to None.
        """
        super().__init__()
        self.residual = residual
    def forward(self, x:torch.Tensor):
        return half_summax(x, self.residual)

def half_mulmax(x:torch.Tensor, residual: Optional[float | torch.Tensor]=None):
    channels = x.size(1)
    if channels == 1: return torch.nn.functional.relu(x, True)
    if channels % 2 == 1: raise ValueError(f"`half_mulmax` got an odd number of channels: {channels = }")
    half = int(channels / 2)
    half1, half2 = x[:,:half], x[:,half:]
    mul = half1 * half2
    if residual is None: return torch.cat((torch.maximum(half1, mul), torch.maximum(half2, mul)), dim=1)
    return torch.cat((torch.maximum(half1, mul), torch.maximum(half2, mul)), dim=1) + x * residual

class HalfMulMax(torch.nn.Module):
    def __init__(self, residual:Optional[float | torch.Tensor]):
        """Splits input into two halves by channels, calculates their product, and returns element-wise maximums of each half with the product.

        Then returns (in pseudocode):

        `channel_concat([elementwise_max(half1, half1 * half2), elementwise_max(half2, half1 * half2)]) + (input * residual)`


        Args:
            residual (Optional[float  |  torch.Tensor]): _description_
        """
        super().__init__()
        self.residual = residual
    def forward(self, x:torch.Tensor):
        return half_mulmax(x, residual=self.residual)

def firstmax(x:torch.Tensor, leak=None):
    channels = x.size(1)
    if channels == 1: return torch.nn.functional.relu(x, inplace = True) if leak is None else torch.nn.functional.leaky_relu(x, inplace=True)
    if leak is None: return torch.maximum(x, x[:,-1].unsqueeze(1))
    else:
        first_channel = x[:,-1].unsqueeze(1)
        return torch.maximum(x, first_channel) + torch.minimum(x, first_channel) * leak

class FirstMax(torch.nn.Module):
    def __init__(self, leak=None):
        """Uses first channel of the input as thresholds for `maximum`.

        Then this is essentially the same as leaky-ReLU with element-wise thresholds.

        Args:
            leak (_type_, optional): Leak. Defaults to None.
        """
        super().__init__()
        self.leak = leak

    def forward(self, x:torch.Tensor):
        return firstmax(x, self.leak)


__all__ = [
    "half_minmax",
    "HalfMinMax",
    "half_summax",
    "HalfSumMax",
    'firstmax',
    "FirstMax"
    ]