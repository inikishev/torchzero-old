from typing import Optional
from collections.abc import Callable
import torch
from torch.optim import Optimizer

from .utils import foreach_param, foreach_group_param
from ..random import rademacher_like

class SPSA(Optimizer):
    def __init__(self, params, lr=1e-2, magn = 1e-3, n = 1, formula='spsa', sampler: Callable = rademacher_like, grad=False, opt=None, ):
        """Simultaneous perturbation stochastic approximation. 
        This generates a small random petrubation `r`, then calculates loss after adding `r` to parameters, and after subtrating `r` from parameters.

        This can function as a gradient estimator for other gradient-based optimizers if `grad` is True,
        otherwise it functions like a standalone optimizer and subtracts the gradients itself.

        The closure is evaluated `n*2` times per step.

        Here is the SPSA gradient formula:
        ```
        loss(parameters + noise) - loss(parameters - noise)
        __________________________________________________
                        2 * noise
        ```

        In RDSA formula is almost the same, except we multiply the top part by noise instead of the bottom.

        Args:
            params (_type_): Parameters to optimize, usually model.parameters().
            lr (_type_, optional): Only takes effect when `grad` is `False`. Step size. Defaults to 1e-2.
            magn (_type_, optional): magnitude of petrubation. Defaults to 1e-5. Many sources recommend decaying this with number of steps.
            n (int, optional): How many random petrubations to generate and average the gradient between them. Defaults to 1.
            formula (str): If `spsa`, uses SPSA formula, if `RDSA`, uses RDSA formula. Defaults to 'spsa'.
            grad (bool, optional): If `False`, this subtracts the gradient multiplied by `lr`, if `True`, this sets gradient to `grad` attribute so that any other gradient based optimizer can make a step. Defaults to False.
            opt (torch.optim.Optimizer, optional): If set and if `grad` is True, makes a step with this optimizer after setting `grad` attribute. Defaults to None.
            sampler (Callable, optional): Sampler for random petrubations. Defaults to rademacher_like.
        """
        defaults = dict(lr=lr, magn=magn, sampler=sampler, grad=grad, formula=formula)
        super().__init__(params, defaults)

        self.n = n
        self.opt = opt


    @torch.no_grad
    def step(self, closure:Callable): # type:ignore #pylint:disable=W0222
        """Performs a single optimization step (parameter update).

        Args:
            closure (Callable): A closure that reevaluates the model and returns the loss. The closure is evaluated `n*2` times per step.
        """

        # save parameters before the step
        for group, p in foreach_group_param(self.param_groups):
            state = self.state[p]
            state['before'] = p.clone()
            state['grad'] = torch.zeros_like(p)

        for step in range(self.n):
            # evaluate +petrubation
            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]

                # generate the petrubation
                state["petrubation"] = group['sampler'](p) * group['magn']

                if step > 0: p.copy_(state['before'])

                p.add_(state['petrubation'])

            # evaluate +petrubation
            lossp = closure()

            # evaluate -petrubation
            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]

                p.copy_(state['before'])

                p.sub_(state['petrubation'])


            lossn = closure()

            # calculate gradient approximation
            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]
                if group['formula'] == 'spsa': state['grad'].add_((lossp - lossn) / (2 * state['petrubation']))
                elif group['formula'] == 'rdsa': state['grad'].add_(((lossp - lossn) * state['petrubation']) / 2)

        # set gradients / do a step
        for group, p in foreach_group_param(self.param_groups):
            state = self.state[p]

            # average gradients
            if self.n > 1: state['grad'].div_(self.n)

            # set grad
            if group['grad']:
                p.copy_(state['before'])
                if p.grad is None: p.grad = state['grad']
                else: p.grad.add_(state['grad'])

            # or make a step
            else: p.copy_(state['before'] - state['grad'] * group['lr'])

        if self.opt is not None: self.opt.step()

        return min(lossp, lossn) # type:ignore

