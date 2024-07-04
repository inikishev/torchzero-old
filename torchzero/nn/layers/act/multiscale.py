from collections.abc import Sequence, Callable
from typing import Optional
from torch import nn
import torch.nn.functional as F
import torch

FloatOrTensor = int | float | torch.Tensor

__all__ = [
    "Multiscale"
]
def _identity(x):return x
class Multiscale(torch.nn.Module):
    def __init__(
        self,
        w_init: Optional[Sequence[float] | torch.Tensor] = (-1., -1/3, 1/3, 1.),
        b_init: Optional[Sequence[float] | torch.Tensor] = (0., 0., 0., 0.,),
        act: Optional[torch.nn.Module | Callable] = None,
        learnable=False,
    ):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(w_init), requires_grad=learnable) if w_init is not None else None
        self.b = torch.nn.Parameter(torch.tensor(b_init), requires_grad=learnable) if b_init is not None else None
        self.act = act if act is not None else _identity

        if self.w is not None and self.b is not None and len(self.w) != len(self.b):
            raise ValueError(f"Length of w and b must be equal, but {len(self.w) = }, {len(self.b) = }")

        self.len = len(self.w) if self.w is not None else len(self.b) if self.b is not None else 0

    def forward(self, x:torch.Tensor):
        if self.len == 0: return x
        repeat_times = [-1 for _ in x.size()] + [self.len]
        permute_axes = list(range(x.ndim))
        permute_axes.insert(1, x.ndim)
        # branching into one-line expressions for performance with unrolled computation
        if self.w is None:
            return self.act((x.unsqueeze(-1).expand(repeat_times) + self.b).permute(permute_axes).flatten(1,2))
        if self.b is None:
            return self.act((x.unsqueeze(-1).expand(repeat_times) * self.w).permute(permute_axes).flatten(1,2))
        return self.act((x.unsqueeze(-1).expand(repeat_times) * self.w + self.b).permute(permute_axes).flatten(1,2))


if __name__ == '__main__':
    inputs = torch.randn(1, 3, 16, 16)
    model = Multiscale(learnable=True)
    y = model(inputs)
    print(y.shape)