from typing import Optional
from collections.abc import Callable
import torch
from torch.optim import Optimizer

from .utils import foreach_param, foreach_group_param

class UpdateToGrad(Optimizer):
    """Convert optimizer update to gradients"""
    def __init__(self, params, opt, lr = 1, mode = "closure", grad_mode = "set", reset=True):
        defaults = dict(
            lr=lr,
            grad_mode=grad_mode,
            reset=reset,
        )
        self.opt = opt
        self.mode = mode.lower()
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure:Optional[Callable] = None): # type:ignore #pylint:disable=W0222
        # save parameters
        for p in foreach_param(self.param_groups):
            state = self.state[p]
            state['before'] = p.clone()

        # do a step
        if self.mode == "step": loss = self.opt.step()
        elif self.mode == "closure": loss = self.opt.step(closure)
        elif callable(self.mode): loss = self.mode(self.opt, closure)
        else: raise ValueError(f"Unknown mode {self.mode}")

        # convert update into gradient
        for group, p in foreach_group_param(self.param_groups):
            state = self.state[p]

            grad_mode = group["grad_mode"]
            lr = group["lr"]
            # set grad to anti-update if None or "set"
            if p.grad is None: p.grad = (state["before"] - p) * lr
            elif grad_mode == "set": p.grad.set_((state["before"] - p) * lr)
            # else add anti-update to grad
            elif grad_mode == "add": p.grad.add_((state["before"] - p) * lr)
            elif callable(grad_mode): grad_mode(p, state["before"], lr)
            else: raise ValueError(f"Unknown grad_mode {grad_mode}")

            # reset params to before update
            if group["reset"]: p.set_(state['before'])

        return loss


class GradChain(Optimizer):
    def __init__(self, params, optimizers, lrs = 1, mode="closure", grad_mode="set", reset=True):
        if isinstance(lrs, (int, float)): lrs = [lrs] * (len(optimizers) - 1)
        if isinstance(mode, (str, Callable)): mode = [mode] * len(optimizers)
        if isinstance(grad_mode, (str, Callable)): grad_mode = [grad_mode] * (len(optimizers) - 1)
        if isinstance(reset, bool): reset = [reset] * (len(optimizers) - 1)

        params = list(params) # to avoid iterating over it with 1st optimizer
        self.chain = [UpdateToGrad(
                params=params,
                lr=lr,
                opt=opt,
                mode=m,
                grad_mode=gm,
                reset=r,
                )
            for lr, opt, m, gm, r in zip(lrs, optimizers[:-1], mode[:-1], grad_mode, reset)]

        self.final_optimizer = optimizers[-1]
        self.final_mode = mode[-1]

        super().__init__(params, {})

    @torch.no_grad
    def step(self, closure:Optional[Callable] = None): # type:ignore #pylint:disable=W0222
        # do step through UpdateToGrad chain
        for opt in self.chain: opt.step(closure)

        # do final optimizer step
        if self.final_mode == "step": loss = self.final_optimizer.step()
        elif self.final_mode == "closure": loss = self.final_optimizer.step(closure)
        elif callable(self.final_mode): loss = self.final_mode(self.final_optimizer, closure)
        else: raise ValueError(f"Unknown mode {self.final_mode}")

        return loss # type:ignore