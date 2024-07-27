from collections.abc import Callable
from typing import Optional
import torch
from torch import optim
from ..utils import get_group_params_tensorlist, get_group_params_and_grads_tensorlist
from .. import _foreach

__all__ = [
    'RandomSearch',
    'RandomAnnealing',
]


class RandomSearch(optim.Optimizer):
    def __init__(
        self,
        params,
        sampler = torch.randn,
        stochastic = False,
        steps_per_sample = 1,
        foreach=True,
        ):
        defaults = dict(sampler=sampler)
        super().__init__(params, defaults)
        self.foreach = foreach
        self.steps_per_sample=steps_per_sample
        self.stochastic=stochastic

        self.lowest_loss = float('inf')

    @torch.no_grad
    def step(self, closure:Callable): # type:ignore # pylint:disable = W0222
        if self.stochastic: self.lowest_loss = closure()

        for step in range(self.steps_per_sample):
            prev = []
            for group in self.param_groups:
                params = get_group_params_tensorlist(group, with_grad=False, foreach=self.foreach)
                prev.append(params.clone())
                params.set_(params.fastfn_like(group['sampler']))

            loss = closure()

            if loss < self.lowest_loss:
                self.lowest_loss = loss

            else:
                for group, prev_params in zip(self.param_groups, prev):
                    params = get_group_params_tensorlist(group, with_grad=False, foreach=self.foreach)
                    params.set_(prev_params)

        return self.lowest_loss

class RandomAnnealing(optim.Optimizer):
    def __init__(
        self,
        params,
        n_steps: Optional[int],
        lr: float = 1,
        sampler=torch.randn,
        stochastic=False,
        steps_per_sample = 1,
        foreach=True,
    ):
        defaults = dict(lr=lr, sampler=sampler, n_steps = n_steps)
        super().__init__(params, defaults)
        self.foreach = foreach
        self.stochastic=stochastic
        self.steps_per_sample = steps_per_sample

        self.lowest_loss = float('inf')

    @torch.no_grad
    def step(self, closure:Callable): # type:ignore # pylint:disable = W0222
        if self.stochastic: self.lowest_loss = closure()

        for step in range(self.steps_per_sample):

            prev = []
            for group in self.param_groups:
                params = get_group_params_tensorlist(group, with_grad=False, foreach=self.foreach)
                prev.append(params.clone())
                params.add_(params.fastfn_like(group['sampler']).mul(group['lr']))

            loss = closure()

            if loss < self.lowest_loss:
                self.lowest_loss = loss

            else:
                for group, prev_params in zip(self.param_groups, prev):
                    params = get_group_params_tensorlist(group, with_grad=False, foreach=self.foreach)
                    params.set_(prev_params)

        for group in self.param_groups:
            if group['n_steps'] is not None: group['lr'] -= group['lr']/group['n_steps']

        return self.lowest_loss
