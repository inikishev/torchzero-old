from typing import Optional
from collections.abc import Callable
import torch
from torch.optim import Optimizer

from .utils import foreach_param, foreach_group_param

class RandomGrad(Optimizer):
    def __init__(self, params, lr=1e-3, opt=None, mul:float = 1, step_opposite=True, sampler: Callable = torch.randn_like):
        defaults = dict(
            lr=lr,
            sampler=sampler,
            mul=mul,
            step_opposite=step_opposite,
        )
        self.opt = opt
        super().__init__(params, defaults)

        self.lowest_loss = float("inf")
        self.n_steps = 0

    @torch.no_grad
    def step(self, closure:Callable): # type:ignore #pylint:disable=W0222
        # get initial loss on 1st step
        # on each step we calculate the loss, make a step, and calculate the loss again
        self.lowest_loss = closure()

        # make a step
        for group, p in foreach_group_param(self.param_groups):
            state = self.state[p]
            # save the params
            state['backup'] = p.clone()
            # create a random direction
            direction = group['sampler'](p, device=p.device) * group["lr"]
            state["direction"] = direction
            # add it
            p.add_(direction)

        # calculate a loss after stepping in the direction
        loss = closure()

        # if loss increased
        if loss > self.lowest_loss:
            # set gradients to anti-step
            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]

                # set the gradients
                if group["step_opposite"]:
                    if p.grad is None: p.grad = state["direction"] * group["mul"]
                    else: p.grad.add_(state["direction"] * group["mul"])

        # if loss decreased
        else:
            self.lowest_loss = loss
            # set gradients to step
            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]

                # set the gradients
                if p.grad is None: p.grad = -state["direction"] * group["mul"]
                else: p.grad.add_(-state["direction"] * group["mul"])

        # restore the params
        for group, p in foreach_group_param(self.param_groups):
            state = self.state[p]
            p.copy_(state['backup'])

        self.n_steps += 1
        if self.opt is not None: self.opt.step()
        return loss

class RandomBestGrad(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        opt=None,
        mul=1,
        bestof=10,
        avgbest=False,
        negate_worst=True,
        sampler: Callable = torch.randn_like,
    ):
        defaults = dict(
            lr=lr,
            sampler=sampler,
            mul=mul,
            avgbest=avgbest,
            negate_worst=negate_worst,
        )
        self.opt = opt
        super().__init__(params, defaults)
        self.bestof = bestof

        self.lowest_loss = float("inf")
        self.n_steps = 0

        self.max_loss_increase = 0
        self.max_loss_decrease = 0


    @torch.no_grad
    def step(self, closure:Callable): # type:ignore #pylint:disable=W0222
        # get baseline loss
        self.lowest_loss = closure()

        # backup params
        for group, p in foreach_group_param(self.param_groups):
            state = self.state[p]
            state['backup'] = p.clone()
            state["nbest"] = 0
            state["best"] = torch.zeros_like(p)

        self.max_loss_increase = 0
        self.max_loss_decrease = 0

        # generate `bestof` directions
        for step in range(self.bestof):
            # make a step
            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]
                # create a random direction
                direction = group['sampler'](p, device=p.device) * group["lr"]
                state["direction"] = direction
                # add it
                p.add_(direction)

            # calculate a loss after stepping in the direction
            loss = closure()

            # if loss increased
            if loss > self.lowest_loss:
                # if loss increased by more than all previous steps
                if loss - self.lowest_loss > self.max_loss_increase:
                    self.max_loss_increase = loss - self.lowest_loss
                    for group, p in foreach_group_param(self.param_groups):
                        # save worst parameters if negate_worst is True
                        if group["negate_worst"]:
                            state = self.state[p]
                            state["worst"] = state["direction"].clone()

            # if loss decreased
            else:
                # if loss decreased by more than all previous steps
                if self.lowest_loss - loss > self.max_loss_decrease:
                    self.max_loss_decrease = self.lowest_loss - loss
                    best = True
                else: best = False
                # save best parameters, or add them for averaging if `avgbest` is True
                for group, p in foreach_group_param(self.param_groups):
                    state = self.state[p]
                    state["nbest"] += 1
                    # add if avgbest
                    if group["avgbest"]:
                        state["best"].add_(state["direction"])
                    # else set if this direction is best so far
                    elif best: state["best"] = state["direction"].clone()

            # restore the params before next step
            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]
                p.copy_(state['backup'])

        # average best, negate if no good
        for group, p in foreach_group_param(self.param_groups):
            state = self.state[p]
            # average good steps
            if group["avgbest"] and state["nbest"] > 0: state["best"].div_(state["nbest"])
            # negate worst if no good steps and negate_worst is True
            if group["negate_worst"] and state["nbest"] == 0: state["best"] = - state["worst"]
            # set the gradients
            if p.grad is None: p.grad = -state["best"] * group["mul"]
            else: p.grad.add_(-state["best"] * group["mul"])

        self.n_steps += 1
        if self.opt is not None: self.opt.step()
        return loss#type:ignore
