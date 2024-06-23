import torch
from ._reduce import _reduce, _ReductionLiterals

def iou(y:torch.Tensor, yhat:torch.Tensor, reduce:_ReductionLiterals=None):
    """
    Intersection over union metric often used for segmentation, also known as Jaccard index.

    y: ground truth in binary one hot format, must be of BC* shape.
    yhat: prediction in binary one hot format, must be of BC* shape.
    reduce: `None`, `"mean"` or `"sum"`.

    returns: vector of len C with iou per each channel, or a single number if reduce is not None.
    """
    y = y.to(torch.bool)
    yhat = yhat.to(torch.bool)
    intersection = (y & yhat).sum((0, *list(range(2, y.ndim))))
    union = (y | yhat).sum((0, *list(range(2, y.ndim))))
    return _reduce(intersection / union, reduce)