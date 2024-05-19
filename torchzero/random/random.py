import random
import torch

def randfill(shape, sampler = random.normalvariate, device=None, requires_grad = False, dtype=None):
    """Return tensor filled with one value sampled from some distribution."""
    return torch.zeros(shape, device=device, requires_grad=requires_grad, dtype=dtype).fill_(sampler(0, 1))

def randfill_like(x:torch.Tensor, sampler = random.normalvariate):
    """Fill tensor with one value sampled from some distribution."""
    return randfill(x.shape, sampler=sampler, device=x.device, requires_grad=x.requires_grad, dtype=x.dtype)

def randrect(shape, fill = randfill_like, device=None, requires_grad = False, dtype=None):
    """Randomly placed rectangle filled with random value from normal distribution."""
    # scalar is just random
    if len(shape) == 0:
        return fill()

    # create start-end slices for the rectangle
    slices = []
    for dim_size in shape:
        if dim_size == 1: slices.append(slice(None))
        else:
            start = random.randrange(0, dim_size-1)
            end = random.randrange(start, dim_size)
            slices.append(slice(start, end))
    # determine fill value
    res = torch.zeros(shape, device=device, requires_grad = requires_grad, dtype=dtype)
    res[slices] = fill(res[slices])
    return res

def randrect_like(x:torch.Tensor, fill = randfill_like):
    """Randomly placed rectangle filled with random value from normal distribution."""
    return randrect(x.shape, fill=fill, device=x.device, requires_grad=x.requires_grad, dtype=x.dtype)

def rademacher(shape, device=None, requires_grad = False, dtype=None):
    """50% to draw a 1 and 50% to draw a -1. Looks like this:
    ```
    [-1,  1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1,  1, -1,  1, -1, -1,  1, -1,  1]
    ```
    """
    return ((torch.rand(shape, dtype=dtype, device=device, requires_grad=requires_grad) < 0.5)) * 2 - 1


def rademacher_like(x:torch.Tensor):
    """50% to draw a 1 and 50% to draw a -1. For example:
    ```
    [-1,  1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1,  1, -1,  1, -1, -1,  1, -1,  1]
    ```
    """
    return rademacher(x.shape, device=x.device, requires_grad=x.requires_grad, dtype=x.dtype)


def similar_like(x:torch.Tensor, sampler=torch.randn_like):
    if x.numel() <= 1: return x.clone()
    mean = x.mean()
    std = x.std()
    if std == 0:
        if mean == 0: return torch.zeros_like(x)
        else: return randfill_like(x, sampler=sampler)
    else:
        rand = sampler(x)
        randmean = rand.mean()
        randstd = rand.std()
        return rand * (std / randstd) + (mean - randmean * randstd / std)