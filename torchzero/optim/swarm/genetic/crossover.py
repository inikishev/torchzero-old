import random
from typing import Optional, Any
from collections.abc import Callable, Sequence
import torch


def crossover_swap(param1:torch.nn.Parameter, param2:torch.nn.Parameter, n=0.5):
    if torch.rand(1) < n: return param2, param1
    return param1, param2

def crossover_uniform_(param1:torch.nn.Parameter, param2:torch.nn.Parameter, n=0.5):
    mask = torch.rand_like(param1) < n
    param1[mask], param2[mask] = param2[mask], param1[mask]
    return param1, param2

def crossover_onepoint_(param1:torch.nn.Parameter, param2:torch.nn.Parameter):
    shape = param1.shape
    if len(shape) != 0:
        # create a slice
        slice1 = []
        slice2 = []
        for dim_size in shape:
            if dim_size == 1:
                slice1.append(slice(None))
                slice2.append(slice(None))
            else:
                start_or_end = random.randrange(0, dim_size-1)
                if torch.rand(1) > 0.5:
                    slice1.append(slice(start_or_end))
                    slice2.append(slice(None, start_or_end))
                else:
                    slice1.append(slice(None, start_or_end))
                    slice2.append(slice(start_or_end))
        # apply the slice
        param1[slice1], param2[slice2] = param2[slice1], param1[slice2]

    return param1, param2

def crossover_mean(param1: torch.nn.Parameter, param2: torch.nn.Parameter):
    return (param1 + param2) / 2

def crossover_random_strat(param1: torch.nn.Parameter, param2: torch.nn.Parameter, strats:Sequence[Callable], weights:Optional[Sequence[float]] = None) -> torch.Tensor | tuple[torch.Tensor]:
    if weights is None: weights = [1 for _ in strats]
    strat = random.choices(strats, weights=weights)[0]
    return strat(param1, param2)