from typing import Optional
from collections.abc import Callable
import torch
from torch.optim import Optimizer

from .utils import foreach_param, foreach_group_param
from ..random import rademacher_like

class OnePointResidualFeeback(Optimizer):
    def __init__(self, params, lr=1e-1, magn=1e-3, sampler: Callable = torch.randn_like, grad=False, opt=None, ):
        """One-point residual feedback - https://arxiv.org/abs/2006.10820

        This gets away with a single closure evaluation per step via taking the difference between current and last loss.

        This can function as a gradient estimator for other gradient-based optimizers if `grad` is True,
        but in my experience it doesn't work too well.
        otherwise it functions like a standalone optimizer and subtracts the gradients itself.

        The closure is evaluated 1 time per step. Authors do claim it works on noisy functions, but I don't know how.

        Args:
            params (_type_): Parameters to optimize, usually model.parameters().
            magn (_type_, optional): Only takes effect when `grad` is `False`. Step size. Defaults to 1e-2.
            grad (bool, optional): If `False`, this subtracts the gradient multiplied by `lr`, if `True`, this sets gradient to `grad` attribute so that any other gradient based optimizer can make a step. Defaults to False.
            opt (torch.optim.Optimizer, optional): If set and if `grad` is True, makes a step with this optimizer after setting `grad` attribute. Defaults to None.
            sampler (Callable, optional): Sampler for random petrubations. Defaults to rademacher_like.
        """
        defaults = dict(magn=magn, sampler=sampler, grad=grad, lr=lr)
        super().__init__(params, defaults)

        self.opt = opt
        self.cur_step = 0
        self.prev_loss = None

    @torch.no_grad
    def step(self, closure:Callable): # type:ignore #pylint:disable=W0222
        """Performs a single optimization step (parameter update).

        Args:
            closure (Callable): A closure that reevaluates the model and returns the loss. The closure is evaluated 1 time per step (two times on 1st step).
        """
        
        if self.cur_step == 0: self.prev_loss = closure()

        # generate petrubation
        for group, p in foreach_group_param(self.param_groups):
            state = self.state[p]
            
            state['before'] = p.clone()

            # generate the petrubation
            state["petrubation"] = group['sampler'](p) * group['magn']

            # add petrubation to parameters
            p.add_(state['petrubation'])

        # evalutate f(x+)
        loss = closure()
        dloss = loss - self.prev_loss


        # set gradients / do a step
        for group, p in foreach_group_param(self.param_groups):
            state = self.state[p]

            # set grad
            if group['grad']:
                p.set_(state['before'])
                if p.grad is None: p.grad = (state['petrubation'] / group['magn']) * dloss
                else: p.grad.add_((state['petrubation'] / group['magn']) * dloss)

            # or make a step
            else:
                # currently petrubation is subbed
                p.sub_(((state['petrubation'] / group['magn']) * dloss) * group['lr'])

        if self.opt is not None: self.opt.step()

        self.prev_loss = loss
        self.cur_step += 1
        return loss # type:ignore

