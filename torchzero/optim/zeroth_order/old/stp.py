from typing import Optional
from collections.abc import Callable
import torch
from torch.optim import Optimizer

from .utils import foreach_param, foreach_group_param
from ..random import rademacher_like

class STP(Optimizer):
    def __init__(self, params, lr=1e-2, sampler: Callable = torch.randn_like, grad=False, opt=None, ):
        """Stochastic Three Points method - https://arxiv.org/pdf/1902.03591.
        
        Generates random petrubation, tries original params, params + petrubation, params - petrubation, and chooses the best

        This can function as a gradient estimator for other gradient-based optimizers if `grad` is True,
        otherwise it functions like a standalone optimizer and subtracts the gradients itself.

        The closure is evaluated `n*3` times per step.

        Args:
            params (_type_): Parameters to optimize, usually model.parameters().
            lr (_type_, optional): Only takes effect when `grad` is `False`. Step size. Defaults to 1e-2.
            grad (bool, optional): If `False`, this subtracts the gradient multiplied by `lr`, if `True`, this sets gradient to `grad` attribute so that any other gradient based optimizer can make a step. Defaults to False.
            opt (torch.optim.Optimizer, optional): If set and if `grad` is True, makes a step with this optimizer after setting `grad` attribute. Defaults to None.
            sampler (Callable, optional): Sampler for random petrubations. Defaults to rademacher_like.
        """
        defaults = dict(lr=lr, sampler=sampler, grad=grad)
        super().__init__(params, defaults)

        self.opt = opt

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

            # generate the petrubation
            state["petrubation"] = group['sampler'](p) * group['lr']

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

        return min(lossp, lossn, losscur) # type:ignore

