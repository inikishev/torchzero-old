from collections.abc import Callable
import torch
from torch.optim import Optimizer

from .utils import foreach_param, foreach_group_param

__all__ = ["RandomSearch", ]
class RandomSearch(Optimizer):
    """Random search. Generates random parameters and saves them if loss decreases."""
    def __init__(self, params, sampler:Callable = torch.randn_like):
        """Random search. Generates random parameters and saves them if loss decreases.

        Args:
            params:
            Usually `model.parameters()`.

            sampler (Callable, optional):
            Sampler that gets passed a parameter and generates random weights of its shape, so functions such as `torch.rand_like`.
            Defaults to torch.randn_like.
        """
        defaults = dict(sampler=sampler)
        super().__init__(params, defaults)

        self.lowest_loss = float("inf")
        self.n_steps = 0

    @torch.no_grad
    def step(self, closure:Callable): # type:ignore #pylint:disable=W0222
        # on first iteration we calculate the initial loss to compare new parameters to on next iterations
        if self.n_steps == 0:
            # save all parameters
            for p in foreach_param(self.param_groups): self.state[p]['best'] = p.clone()
            # evaluate the initial loss
            self.lowest_loss = closure()

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
    """Random shrinking search. Generates random parameters in some area around the best parameter, and saves them if loss decreases.
    Area size depends on `lr` parameter. A simple linear decay to 0 scheduler will be used if `nsteps` is provided; otherwise use any other scheduler."""
    def __init__(self, params, lr, nsteps = None, sampler:Callable = torch.randn_like):
        """Random search. Generates random parameters and saves them if loss decreases.

        Args:
            params:
            Usually `model.parameters()`.

            sampler (Callable, optional):
            Sampler that gets passed a parameter and generates random weights of its shape, so functions such as `torch.rand_like`.
            Defaults to torch.randn_like.
        """
        defaults = dict(lr=lr, nsteps=nsteps, sampler=sampler)
        super().__init__(params, defaults)

        self.lowest_loss = float("inf")
        self.cur_step = 0

    @torch.no_grad
    def step(self, closure:Callable): # type:ignore #pylint:disable=W0222
        # on first iteration we calculate the initial loss to compare new parameters to on next iterations
        if self.cur_step == 0:
            # save all parameters
            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]
                state['best'] = p.clone()
                state['init lr'] = group['lr']
            # evaluate the initial loss
            self.lowest_loss = closure()

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