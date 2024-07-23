from typing import Literal
import torch

_ReductionLiterals = Literal['mean', 'sum', None, 'none', 'nan0', 'nan1']

def _reduce(x:torch.Tensor, reduction: _ReductionLiterals):
    if reduction is None or reduction == 'none': return x
    reduction = reduction.lower() # type:ignore
    if reduction == 'nan0': return torch.nan_to_num(x, 0)
    if reduction == 'nan1': return torch.nan_to_num(x, 1)
    if reduction == 'mean': return torch.mean(x)
    if reduction == 'sum': return torch.sum(x)
    else: raise ValueError(f'Invalid reduction: {reduction}')