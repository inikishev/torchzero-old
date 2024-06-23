from collections.abc import Callable, Sequence
from typing import Optional
import torch

class BinaryToMulticlass(torch.nn.Module):
    """
    Computes a binary loss for a multiclass problem per each channels weighted by `weights`.

    Expects a `BC*` tensor, where `B` is the batch size, `C` is the channel in one-hot encoded array, and `*` is any number of dimensions.

    The original loss must take a `B*` tensor, probably with probability values in 0-1 range.
    """
    def __init__(self, loss:torch.nn.Module | Callable, weights:Optional[Sequence[int | float]] = None):
        super().__init__()
        self.loss = loss
        if weights is None: self.weights = [1 for _ in range(len(self.base))]
        else: self.weights = weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_channels = torch.unbind(pred, 1)
        target_channels = torch.unbind(target, 1)
        loss = 0
        for pch, tch, w in zip(pred_channels, target_channels, self.weights):
            loss += self.loss(pch, tch) * w
        return loss # type:ignore


class Convert2DLossTo3D(torch.nn.Module):
    """
    Converts a 2d loss to a 3d loss, where the last dimension is the channel dimension.

    The original loss must accept a `BC**` tensor, i.e. batch-channel-2D.

    Forward therefore expects a `BC***` (batch-channel-3D) tensor, which will be unpacked into a `BC**` tensor by unrolling 1st dim.
    """
    def __init__(self, loss:torch.nn.Module | Callable):
        super().__init__()
        self.loss = loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred.view(pred.shape[0],pred.shape[1],  pred.shape[2]*pred.shape[3], pred.shape[4]),
                         target.view(target.shape[0], target.shape[1],  target.shape[2]*target.shape[3], target.shape[4]))

