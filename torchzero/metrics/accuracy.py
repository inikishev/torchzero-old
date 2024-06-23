import torch
from ._reduce import _reduce, _ReductionLiterals

def accuracy(y:torch.Tensor, yhat:torch.Tensor, reduce:_ReductionLiterals = "mean"):
    """
    Accuracy metric. Defined as number of correct predictions divided by total number of predictions.

    y: ground truth in binary one hot format, any shape.
    yhat: prediction in binary one hot format, any shape.
    reduce: `None`, `"mean"` or `"sum"`.

    returns: vector of len C with accuracy per each channel, or a single number if reduce is not None.
    """
    return _reduce((y == yhat).mean((0, *list(range(2, y.ndim)))), reduce)