from typing import Optional
from collections.abc import Callable
import torch
from torch.optim import Optimizer

from .utils import foreach_param, foreach_group_param

__all__ = ["RandomSearch", ]
class RandomSearch(Optimizer):
    def __init__(self, params, sampler:Callable = torch.randn_like):
        """Random search. Generates random parameters and saves them if loss decreases.

        Random search is often used for finding hyperparameters,
        or to quickly find the minimum of some cheap low-dimensional problem without having to worry about more sophisticated optimization methods.

        This evaluates closure once per step.

        Args:
            params: Parameters to optimize. Usually `model.parameters()`.

            sampler (Callable, optional): Sampler that gets passed a parameter and generates random weights of its shape, so functions such as `torch.randn_like`. Defaults to torch.randn_like. Supports parameter groups.
        """
        defaults = dict(sampler=sampler)
        super().__init__(params, defaults)

        self.lowest_loss = float("inf")
        self.n_steps = 0

        # save all parameters
        for p in foreach_param(self.param_groups): self.state[p]['best'] = p.clone()
        # evaluate the initial loss

    @torch.no_grad
    def step(self, closure:Callable): # type:ignore #pylint:disable=W0222
        """Performs a single optimization step (parameter update).

        Args:
            closure (Callable): A closure that reevaluates the model and returns the loss. The closure is evaluated twice on the first step, and then once per step.
        """
        # on first iteration we calculate the initial loss to compare new parameters to on next iterations
        if self.n_steps == 0: self.lowest_loss = closure()

        # make a step
        for group, p in foreach_group_param(self.param_groups):

            # set parameter group to new random values
            p.set_(group['sampler'](p, device=p.device))

        # calculate loss
        loss = closure()

        # if loss is not lower than the lowest loss, restore the best parameters
        if loss > self.lowest_loss:
            for p in foreach_param(self.param_groups):
                state = self.state[p]
                p.set_(state['best'].clone())

        # if loss is lower
        else:
            # save new lowest loss
            self.lowest_loss = loss
            # save new best parameters
            for p in foreach_param(self.param_groups): self.state[p]['best'] = p.clone()

        self.n_steps += 1
        return self.lowest_loss

class RandomShrinkingSearch(Optimizer):
    def __init__(self, params, lr: float = 1, nsteps:Optional[int] = None, sampler:Callable = torch.randn_like, stochastic=False):
        """Random shrinking search. Generates random parameters in a shrinking area around the best found parameters.

        Search area size depends on `lr` parameter.
        Linear decay from 1 to 0 will be used if `nsteps` is provided; otherwise use any other lr scheduler.
        #### If you neither specify `nsteps` nor use an lr scheduler, this will be equivalent to normal random search.

        This evaluates closure once per step.

        Args:
            params: Parameters to optimize. Usually `model.parameters()`.

            lr (float): random values from `sampler` will be multiplied by this, i.e. lower values mean smaller search area. If `nsteps` is specified, this linearly will decay to 0. Defaults to 1. Supports parameter groups and lr schedulers.

            nsteps (int): Number of steps for linear lr decay. One step is performed each time `step` is called. Without `nsteps`/lr scheduler this will be equivalent to normal random search.

            sampler (Callable, optional): Sampler that gets passed a parameter and generates random weights of its shape, so functions such as `torch.rand_like`. Defaults to torch.randn_like. Supports parameter groups.
        """
        defaults = dict(lr=lr, nsteps=nsteps, sampler=sampler)
        super().__init__(params, defaults)

        self.stochastic = stochastic

        self.lowest_loss = float("inf")
        self.cur_step = 0

        # save all parameters
        for group, p in foreach_group_param(self.param_groups):
            state = self.state[p]
            state['best'] = p.clone()
            state['init lr'] = group['lr']
        # evaluate the initial loss

    @torch.no_grad
    def step(self, closure:Callable): # type:ignore #pylint:disable=W0222
        """Performs a single optimization step (parameter update).

        Args:
            closure (Callable): A closure that reevaluates the model and returns the loss. The closure is evaluated twice on the first step, and then once per step.
        """
        # on first iteration we calculate the initial loss to compare new parameters to on next iterations
        if self.cur_step == 0 or self.stochastic: self.lowest_loss = closure()

        # make a step
        for group, p in foreach_group_param(self.param_groups):
            state = self.state[p]
            # set parameter group to new random values around current best parameters
            p.set_(state['best'] + group['sampler'](p, device=p.device) * group['lr'])

            # decay lr
            if group['nsteps'] is not None:
                group['lr'] = state['init lr'] * ((group['nsteps'] - self.cur_step) / group['nsteps'])

        # calculate loss
        loss = closure()

        # if loss is not lower than the lowest loss, restore the best parameters
        if loss > self.lowest_loss:
            for p in foreach_param(self.param_groups):
                state = self.state[p]
                p.set_(state['best'].clone())

        # if loss is lower
        else:
            # save new lowest loss
            self.lowest_loss = loss
            # save new best parameters
            for p in foreach_param(self.param_groups): self.state[p]['best'] = p.clone()

        self.cur_step += 1
        return self.lowest_loss