"losses"
from collections.abc import Callable, Sequence
from typing import Optional
import torch
from ..nn.layers.sequential import ensure_module

class LossMean(torch.nn.Module):
    def __init__(self, losses:Sequence[torch.nn.Module | Callable], weights:Optional[Sequence[int|float]] = None):
        """Takes optionally weighted mean of the losses.

        Args:
            losses (Sequence[torch.nn.Module  |  Callable]): _description_
            weights (Optional[Sequence[int | float]], optional): _description_. Defaults to None.
        """
        super().__init__()
        self.losses = torch.nn.ModuleList([ensure_module(i) for i in losses])
        if weights is not None: self.weights = [i / sum(weights) for i in weights]
        else: weights = [1/len(losses) for _ in range(len(losses))]

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        loss = 0
        for loss_func, weight in zip(self.losses, self.weights):
            loss += loss_func(y_true, y_pred) * weight
        return loss # type:ignore

class LossSum(torch.nn.Module):
    def __init__(self, losses:Sequence[torch.nn.Module | Callable], weights:Optional[Sequence[int|float]] = None):
        """Takes optionally weighted sum of losses.

        Args:
            losses (Sequence[torch.nn.Module  |  Callable]): _description_
            weights (Optional[Sequence[int | float]], optional): _description_. Defaults to None.
        """
        super().__init__()
        self.losses = torch.nn.ModuleList([ensure_module(i) for i in losses])
        if weights is None: self.weights = [1 for _ in range(len(losses))]
        else: self.weights = weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0
        for loss_func, weight in zip(self.losses, self.weights):
            loss += loss_func(pred, target) * weight
        return loss # type:ignore
