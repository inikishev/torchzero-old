from collections.abc import Callable
import torch
from torch.func import jacrev # type:ignore

def newtons_method(f:Callable, x0:torch.Tensor, n:int):
    """Find root of scalar function `f` starting from initial guess `x0` using Newton's method for `n` iterations."""
    df = jacrev(f) # derivative of f
    # for n iterations
    for iteration in range(n):
        print(iteration, x0)
        # apply the newtons method formula to obtain a new, better guess
        x0 = x0 - (f(x0) / df(x0))
    return x0

def newtons_optimization(f:Callable, x0:torch.Tensor, n:int):
    """Find root of the derivative of scalar function `f` starting from initial guess `x0` using Newton's method for `n` iterations."""
    df = jacrev(f) # 1st order derivative
    d2f = jacrev(df) # 2nd order derivative
    # for n iterations
    for iteration in range(n):
        print(iteration, x0)
        # apply the newtons method formula to obtain a new, better guess
        x0 = x0 - (df(x0) / d2f(x0)) # type:ignore
    return x0