from collections.abc import Callable
import math
import torch
from torch.func import jacrev # type:ignore

def taylor(f, a, order):
    """Construct a new function, an `order`th order taylor approximation of scalar function `f` around `a`. 
    Note that this implementation is probably highly inefficient, but I am not sure."""
    series_fns = [f]
    factorials = [0] # first value won't be used

    for i in range(order):
        f = jacrev(f)  # get next order derivative of f
        series_fns.append(f)
        factorials.append(math.factorial(i+1))

    def evaluate_taylor(x, order = order):
        res = series_fns[0](a) # first element, f(a)
        if order > 0:
            for n in range(1, order+1):
                res += series_fns[n](a) / factorials[n] * (x - a) ** n
        return res

    return evaluate_taylor