import torch
from ._reduce import _reduce, _ReductionLiterals
def dice(y:torch.Tensor, yhat:torch.Tensor, reduction:_ReductionLiterals='nan1'):
    """
    Sørensen–Dice coefficient often used for segmentation. Defined as two intersections over sum. Equivalent to F1 score.

    y: ground truth in one hot format, must be of BC(*) shape.
    yhat: prediction in one hot format, must be of BC(*) shape.
    reduce: `None`, `"mean"` or `"sum"`.

    returns: vector of len C with dice per each channel, or a single number if reduce is not None.
    """
    y = y.to(torch.bool)
    yhat = yhat.to(torch.bool)
    intersection = (y & yhat).sum((0, *list(range(2, y.ndim))))
    sum_ = y.sum((0, *list(range(2, y.ndim)))) + yhat.sum((0, *list(range(2, y.ndim))))
    return _reduce((2*intersection) / sum_, reduction)


def softdice(y:torch.Tensor, yhat:torch.Tensor, reduction:_ReductionLiterals='nan1'):
    """
    A differentiable verison of Sørensen–Dice coefficient, uses multiplication instead of intersection.

    y: ground truth in one hot format, must be of BC(*) shape. All vectors along C must sum to 1 (so apply softmax).
    yhat: prediction in one hot format, must be of BC(*) shape. All vectors along C must sum to 1 (so apply softmax).
    reduce: `None`, `"mean"` or `"sum"`.

    returns: vector of len C with soft dice per each channel, or a single number if reduce is not None.
    """
    intersection = (y * yhat).sum((0, *list(range(2, y.ndim))))
    sum_ = y.sum((0, *list(range(2, y.ndim)))) + yhat.sum((0, *list(range(2, y.ndim))))
    return _reduce((2*intersection) / sum_, reduction)
