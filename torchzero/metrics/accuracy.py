import torch
from ._reduce import _reduce, _ReductionLiterals

def accuracy(y:torch.Tensor, yhat:torch.Tensor, reduction:_ReductionLiterals = "nan1"):
    """
    Accuracy metric. Defined as number of correct predictions divided by total number of predictions.

    y: ground truth in binary one hot format, must be of BC(*) shape.
    yhat: prediction in binary one hot format, must be of BC(*) shape.
    reduce: `None`, `"mean"` or `"sum"`.

    returns: vector of len C with accuracy per each channel, or a single number if reduce is not None.
    """
    return _reduce((y == yhat).float().mean((0, *list(range(2, y.ndim)))), reduction)