from typing import Optional
from collections.abc import Callable
import torch
from torch.optim import Optimizer

from .utils import foreach_param, foreach_group_param
from ..random import rademacher_like

class TwoStepRS(Optimizer):
    def __init__(self, params, lr=1e-2, lr2=1e-1, sampler: Callable = torch.randn_like, grad=False, opt=None, ):
        """Two-step random search - https://arxiv.org/pdf/2110.13265.

        Generates random petrubation, on every first step tries original params, params + petrubation * lr1, params - petrubation * lr1, and chooses the best.
        On every second step tries original params, params + petrubation * lr2, params - petrubation * lr2, and chooses the best.
        So it alternates between two lrs, one smaller and one bigger.

        This can function as a gradient estimator for other gradient-based optimizers if `grad` is True,
        otherwise it functions like a standalone optimizer and subtracts the gradients itself.

        However it doesn't really make sense as a gradient estimator, so that probably won't work well.

        The closure is evaluated `n*3` times per step.

        Args:
            params (_type_): Parameters to optimize, usually model.parameters().
            lr (_type_, optional): Only takes effect when `grad` is `False`. Step size. Defaults to 1e-2.
            grad (bool, optional): If `False`, this subtracts the gradient multiplied by `lr`, if `True`, this sets gradient to `grad` attribute so that any other gradient based optimizer can make a step. Defaults to False.
            opt (torch.optim.Optimizer, optional): If set and if `grad` is True, makes a step with this optimizer after setting `grad` attribute. Defaults to None.
            sampler (Callable, optional): Sampler for random petrubations. Defaults to rademacher_like.
        """
        defaults = dict(lr=lr, lr2=lr2, sampler=sampler, grad=grad)
        super().__init__(params, defaults)

        self.opt = opt
        self.cur_step = 0

    @torch.no_grad
    def step(self, closure:Callable): # type:ignore #pylint:disable=W0222
        """Performs a single optimization step (parameter update).

        Args:
            closure (Callable): A closure that reevaluates the model and returns the loss. The closure is evaluated `n*2` times per step.
        """

        # evaluate f(xk)
        losscur = closure()

        # generate f(x+)
        for group, p in foreach_group_param(self.param_groups):
            state = self.state[p]

            state['before'] = p.clone()

            if self.cur_step % 2 == 0: lr = group['lr']
            else: lr = group['lr2']

            # generate the petrubation
            state["petrubation"] = group['sampler'](p) * lr

            # add petrubation to parameters
            p.add_(state['petrubation'])

        # evalutate f(x+)
        lossp = closure()

        # generate f(x-)
        for group, p in foreach_group_param(self.param_groups):
            state = self.state[p]

            # sub petrubation
            p.sub_(state['petrubation'] * 2)

        # evalutate f(x-)
        lossn = closure()

        # set gradients / do a step
        for group, p in foreach_group_param(self.param_groups):
            state = self.state[p]

            # set grad
            if group['grad']:
                p.set_(state['before'])
                if lossp <= lossn and lossp <= losscur:
                    if p.grad is None: p.grad = -state['petrubation'] / group['lr']
                    else: p.grad.add_(-state['petrubation'] / group['lr'])
                elif lossn <= lossp and lossn <= losscur:
                    if p.grad is None: p.grad = state['petrubation'] / group['lr']
                    else: p.grad.add_(state['petrubation'] / group['lr'])
                else:
                    if p.grad is None: p.grad = torch.zeros_like(p)

            # or make a step
            else:
                # currently petrubation is subbed
                if lossp <= lossn and lossp <= losscur: p.add_(state['petrubation'] * 2)
                elif losscur <= lossp and losscur <= losscur: p.set_(state['before'])

        if self.opt is not None: self.opt.step()

        self.cur_step += 1
        return min(lossp, lossn, losscur) # type:ignore

