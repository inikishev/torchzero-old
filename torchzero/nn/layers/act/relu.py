from typing import Optional
from torch import nn
import torch.nn.functional as F
import torch

FloatOrTensor = int | float | torch.Tensor


__all__ = [
    "general_relu",
    'GeneralReLU',
    'symmetric_relu',
    'SymReLU',
    "LearnReLU",
    'LearnLeakyReLU',
    'LearnGeneralReLU',
    'LearnSymReLU',
    "general_relu",
    "symmetric_relu",
]


def general_relu(x, leak:Optional[float] = None, sub:Optional[float] = None, maxv: Optional[FloatOrTensor] = None, inplace=True):
    """GeneralReLU described by Jeremy Howard in his fastai courses. Leaky ReLU but shifted down.

    Args:
        leak (_type_, optional): _description_. Defaults to None.
        sub (_type_, optional): _description_. Defaults to None.
        maxv (Optional[float], optional): _description_. Defaults to None.
    """
    if isinstance(maxv, (int,float)): maxv = torch.tensor(maxv)
    x = F.leaky_relu(x, leak, inplace=inplace) if leak else F.relu(x, inplace=inplace)
    if sub is not None: x -= sub
    if maxv is not None: x = x.minimum(maxv)
    return x

class GeneralReLU(nn.Module):
    def __init__(self, leak:Optional[float] = None, sub:Optional[float] = None, maxv: Optional[FloatOrTensor] = None, inplace=True):
        """GeneralReLU described by Jeremy Howard in his fastai courses. Leaky ReLU but shifted down.

        Args:
            leak (_type_, optional): _description_. Defaults to None.
            sub (_type_, optional): _description_. Defaults to None.
            maxv (Optional[float], optional): _description_. Defaults to None.
        """
        super().__init__()
        if isinstance(maxv, (int,float)): self.maxv = torch.tensor(maxv)
        elif maxv is None: self.maxv = None
        self.leak, self.sub = leak, sub
        self.inplace = inplace


    def forward(self, x):
        return general_relu(x, leak = self.leak, sub = self.sub, maxv = self.maxv, inplace=self.inplace)

def symmetric_relu(x, leak:float = 0.2, sub:float = 1., maxv: Optional[FloatOrTensor] = None, inplace=True):
    """Symmetric version of leaky ReLU.

    Args:
        leak (float, optional): _description_. Defaults to 0.2.
        sub (float, optional): _description_. Defaults to 1.
        maxv (Optional[float], optional): _description_. Defaults to None.
    """
    if isinstance(maxv, (int, float)): maxv = torch.tensor(maxv)
    x = F.leaky_relu(x+sub, leak, inplace=inplace) - sub
    x = -F.leaky_relu(-x + sub, leak, inplace=inplace) + sub
    if maxv is not None: x = x.minimum(maxv)
    return x

class SymReLU(nn.Module):
    def __init__(self, leak: float = 0.2, sub: float = 1, maxv: Optional[FloatOrTensor] = None, inplace=True):
        """Symmetric version of leaky ReLU.

        Args:
            leak (float, optional): _description_. Defaults to 0.2.
            sub (float, optional): _description_. Defaults to 1.
            maxv (Optional[float], optional): _description_. Defaults to None.
        """
        super().__init__()
        if isinstance(maxv, (int,float)): self.maxv = torch.tensor(maxv)
        elif maxv is None: self.maxv = None
        self.leak, self.sub = leak, sub
        self.inplace = inplace

    def forward(self, x: torch.Tensor):
        return symmetric_relu(x, leak = self.leak, sub = self.sub, maxv = self.maxv, inplace=self.inplace)

class LearnReLU(nn.Module):
    def __init__(self):
        """ReLU with a learnable threshold.
        """
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x: torch.Tensor):
        return torch.maximum(x, self.threshold)

class LearnLeakyReLU(nn.Module):
    def __init__(self, leak_init = 0.1):
        """Leaky ReLU with learnable threshold and leak.

        Args:
            leak_init (float, optional): _description_. Defaults to 0.1.
        """
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.leak = nn.Parameter(torch.tensor(leak_init), requires_grad=True)

    def forward(self, x: torch.Tensor):
        return torch.maximum(x, self.threshold) + torch.minimum(x, self.threshold) * self.leak

class LearnGeneralReLU(nn.Module):
    def __init__(self, leak_init = 0.1, sub_init = 0.1, maxv_init:Optional[float] = 10.):
        """GeneralReLU described by Jeremy Howard in his fastai courses, but also learnable.
        This is calculated as `(LeakyReLU(x, threshold = W1, leak=W2) - W3).clamp(min=W4, max=W5)`. All W's are learnable.

        Args:
            leak_init (float, optional): _description_. Defaults to 0.1.
            sub_init (float, optional): _description_. Defaults to 0.1.
            maxv_init (Optional[float], optional): _description_. Defaults to 10..
        """
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.leak = nn.Parameter(torch.tensor(leak_init), requires_grad=True)
        self.sub = nn.Parameter(torch.tensor(sub_init), requires_grad=True)
        if maxv_init is not None: self.maxv = nn.Parameter(torch.tensor(maxv_init), requires_grad=True)
        else: self.maxv = None

    def forward(self, x: torch.Tensor):
        x = (torch.maximum(x, self.threshold) + torch.minimum(x, self.threshold) * self.leak) - self.sub
        if self.maxv is not None: x = x.minimum(self.maxv)
        return x

class LearnSymReLU(nn.Module):
    def __init__(self, lower_leak_init = 0.1, upper_leak_init = 0.1, sub_init = 0.1, maxv_init:Optional[float] = 10., minv_init:Optional[float] = -10.):
        """Symmetric version of leaky ReLU with learnable threshold, lower and upper leaks, and vertical position and min/max values.

        Args:
            leak_init (float, optional): _description_. Defaults to 0.1.
            sub_init (float, optional): _description_. Defaults to 0.1.
            maxv_init (Optional[float], optional): _description_. Defaults to 10..
            minv_init (Optional[float], optional): _description_. Defaults to -10.
        """
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.lower_leak = nn.Parameter(torch.tensor(float(lower_leak_init)), requires_grad=True)
        self.upper_leak = nn.Parameter(torch.tensor(float(upper_leak_init)), requires_grad=True)
        self.sub = nn.Parameter(torch.tensor(float(sub_init)), requires_grad=True)

        if maxv_init is not None: self.maxv = nn.Parameter(torch.tensor(float(maxv_init)), requires_grad=True)
        else: self.maxv = None
        if minv_init is not None: self.minv = nn.Parameter(torch.tensor(float(minv_init)), requires_grad=True)
        else: self.minv = None


    def forward(self, x: torch.Tensor):
        # lower part
        x = (torch.maximum(x, self.threshold) + torch.minimum(x, self.threshold) * self.lower_leak) - self.sub
        # upper part
        x = -((torch.maximum(-x, self.threshold) + torch.minimum(-x, self.threshold) * self.upper_leak) - self.sub)
        if self.maxv is not None: x = x.minimum(self.maxv)
        if self.minv is not None: x = x.maximum(self.minv)
        return x

