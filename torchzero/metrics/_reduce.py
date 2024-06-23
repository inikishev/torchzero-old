from typing import Literal
import torch

_ReductionLiterals = Literal['mean', 'sum', None]

def _reduce(x:torch.Tensor, reduction: _ReductionLiterals):
    if reduction is None: return x
    reduction = reduction.lower() # type:ignore
    if reduction == 'mean': return torch.mean(x)
    elif reduction == 'sum': return torch.sum(x)
    else: raise ValueError(f'Invalid reduction: {reduction}')