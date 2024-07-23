from collections.abc import Callable
from typing import Optional, Literal

import torch
from torch import optim
from ..utils import get_group_params, get_group_params_and_grads
from .. import _foreach

__all__ = [
    "FDSA",
]
class FDSA(optim.Optimizer):
    def __init__(
        self,
        params,
        lr:Optional[float] = 1e-3,
        magn:float = 1e-3,
        set_grad = False,
        opt = None,
    ):
        """Finite Differences Stochastic Approximation (FDSA).

        Args:
            params (_type_): _description_
            lr (Optional[float], optional): _description_. Defaults to 1e-3.
            magn (float, optional): _description_. Defaults to 1e-5.
            set_grad (bool, optional): _description_. Defaults to False.
            opt (_type_, optional): _description_. Defaults to None.
            foreach (bool, optional): _description_. Defaults to True.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """
        if lr is None and magn is None: raise ValueError("Either lr or magn must be specified.")
        if magn is None: magn = lr
        if set_grad is False and opt is not None: raise ValueError("opt can only be used with set_grad=True")
        if lr is None and not set_grad: raise ValueError("lr must be specified if set_grad=False")

        defaults = dict(lr=lr, magn=magn)
        super().__init__(params, defaults)

        self.set_grad = set_grad
        self.opt = opt

    @torch.no_grad
    def step(self, closure:Callable):# pylint:disable=W0222 # type:ignore

        for group in self.param_groups:
            # this creates gradients if they are None
            params, grads = get_group_params_and_grads(group, with_grad=False, create_grad=self.set_grad)
            for p in params:
                # get parameter vector
                vec = p.ravel()
                # construct gradient vector
                grad = torch.zeros_like(vec)
                # for each scalar in the vector
                for idx in range(len(vec)):
                    # positive loss
                    vec[idx] += group['magn']
                    loss_pos = closure()
                    # negative loss
                    vec[idx] -= 2 * group['magn']
                    loss_neg = closure()
                    # restore original value
                    vec[idx] += group['magn']
                    # calculate gradient
                    grad[idx] = (loss_pos - loss_neg) / (2 * group['magn'])

                # set grad attr
                if self.set_grad: p.grad.add_(grad.view_as(p)) # type:ignore
                # or apply grad
                else: p.sub_(grad.view_as(p), alpha=group['lr'])

        if self.opt is not None: self.opt.step()
        return loss_pos # type:ignore