from typing import Optional
from collections.abc import Callable
import torch
from torch.optim import Optimizer

from .utils import foreach_param, foreach_group_param
from ..random import rademacher_like


class RandomOptimizer(Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-3,
        navg = 1,
        nsteps = 1,
        opposite = True,
        grad = False,
        sampler: Callable = torch.randn_like,
        stochastic = True,
        opt = None,
        ):
        defaults = dict(
            lr = lr,
            opposite = opposite,
            grad = grad,
            sampler = sampler,
        )
        super().__init__(params, defaults)
        self.stochastic = stochastic
        self.opt = None
        self.nsteps = nsteps
        self.navg = navg

        self.lowest_loss = float("inf")
        self.cur_step = 0

    @torch.no_grad
    def step(self, closure:Callable): # type:ignore #pylint:disable=W0222
        """Performs a single optimization step (parameter update).

        Args:
            closure (Callable): A closure that reevaluates the model and returns the loss. The closure is evaluated `navg * nsteps * 2` times per step.
        """


        if self.nsteps > 1 and self.opt is None:
            # save initial params
            for group, p in foreach_group_param(self.param_groups):
                if group['grad']:
                    state = self.state[p]
                    state['before closure steps'] = p.clone()

        for closurestep in range(self.nsteps):
            # evaluate loss before a step
            if (self.stochastic or self.cur_step == 0) and closurestep == 0:
                self.lowest_loss = closure()

            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]
                state['petrubation'] = group['sampler'](p)

                # set gradients / do a step
            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]

                # average gradients
                if self.navg > 1: state['grad'].div_(self.navg)

                # set grad
                if group['grad'] and (closurestep == self.nsteps - 1 or self.opt is not None):
                    p.copy_(state['before'])
                    if p.grad is None: p.grad = state['grad']
                    else: p.grad.add_(state['grad'])

                # or make a step
                else: p.copy_(state['before'] - state['grad'] * group['lr'])

            if self.opt is not None: self.opt.step()

        if self.nsteps > 1 and self.opt is None:
            # load initial params
            for group, p in foreach_group_param(self.param_groups):
                if group['grad']:
                    state = self.state[p]
                    p.copy_(state['before closure steps'])

        return self.lowest_loss

