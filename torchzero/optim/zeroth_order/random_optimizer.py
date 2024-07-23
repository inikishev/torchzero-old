from collections.abc import Callable
from typing import Optional
import torch
from torch import optim
from ..utils import get_group_params_tensorlist, get_group_params_and_grads_tensorlist
from .. import _foreach

__all__ = [
    "RandomOptimizer",
]
class RandomOptimizer(optim.Optimizer):
    def __init__(
        self,
        params,
        lr:Optional[float] = 1e-3,
        magn:Optional[float] = None,
        step_back = True,
        sampler = torch.randn,
        set_grad = False,
        opt = None,
        stochastic = True,
        steps_per_sample = 1,
        foreach = True,
        fast_sampling = False,
    ):
        """Random optimizer. Tries a step in a random direction, and goes there if loss decreases.

        Args:
            params (_type_): Iterable of parameters to optimize. Usually `model.parameters()`.
            lr (_type_): Learning rate, also used as magnitude if `magn` is None. This has no effect if `set_grad` is True.
            magn (_type_): Random direction magnitude. If `None`, this is equal to `lr` and will be slightly more computationally efficient.
            step_back (bool, optional): Whether to step in the opposite direction if loss increased, otherwise undo the step. Defaults to True.
            set_grad (bool, optional): If `True`, this acts like a gradient approximator and sets `grad` attribute, otherwise this acts as a standalone optimizer and doesn't touch `grad` attribute. Defaults to False.
            opt (_type_, optional): Do a step with any gradient-based optimizer after approximating gradients. Defaults to None.
            stochastic (bool, optional): If `True`, evaluates loss each time before doing a random step. Defaults to True.
            steps_per_sample (int, optional): Does multiple steps per sample, which is efficient because it does one initial evaluation + only one evaluation per step. Not useful if `stochastic` is False. Defaults to 1.
            foreach (bool, optional): Use `foreach` operations which makes it much faster (the random generator will generate different tensors with and without this!). Defaults to True.
        """
        if lr is None and magn is None: raise ValueError("Either lr or magn must be specified.")
        if magn is None: magn = lr
        if set_grad is False and opt is not None: raise ValueError("opt can only be used with set_grad=True")

        defaults = dict(lr=lr, magn=magn, step_back=step_back, sampler=sampler)
        super().__init__(params, defaults)

        self.foreach = foreach
        self.fast_sampling = fast_sampling
        self.set_grad = set_grad
        self.opt = opt
        self.stochastic = stochastic
        self.steps_per_sample = steps_per_sample

        self.lowest_loss = float('inf')

    @torch.no_grad
    def step(self, closure:Callable): # pylint:disable=W0222 # type:ignore
        # if stochastic, evaluate  loss on each step
        if self.stochastic: self.lowest_loss = closure()

        # we can do multiple steps per sample, only evaluating once per step
        for step in range(self.steps_per_sample):

            # create petrubations and apply them
            petrubations_per_group: list[_foreach.TensorList] = []
            for idx, group in enumerate(self.param_groups):
                params = get_group_params_tensorlist(group, with_grad=False, foreach=self.foreach)
                group_petrubation = params.fastfn_like(group['sampler'], reuse = self.fast_sampling)
                petrubations_per_group.append(group_petrubation)
                params.add_(group_petrubation, alpha = group['magn'])

            # evaluate new loss
            loss = closure()

            # loss is lower
            if loss < self.lowest_loss:
                self.lowest_loss = loss
                # undo the petrubation and set grad attribute if set_grad
                for group, petrubation in zip(self.param_groups, petrubations_per_group):
                    if self.set_grad:
                        params, grads = get_group_params_and_grads_tensorlist(group, with_grad=False, foreach=self.foreach)
                        # undo the petrubation
                        params.sub_(petrubation, alpha=group['magn'])
                        # accumulate gradients
                        grads.sub_(petrubation)

                    # if lr and magnitude are different, we subtract petrubation * (magn - lr) to convert magn to lr.
                    elif group['lr'] != group['magn']:
                        params = get_group_params_tensorlist(group, with_grad=False, foreach=self.foreach)
                        params.sub_(petrubation, alpha=group['magn'] - group['lr'])

                    # else we don't do anything, params are already updated with correct lr = magn

            # loss is bigger
            else:
                # undo the petrubation
                for group, petrubation in zip(self.param_groups, petrubations_per_group):

                    # undo the petrubation and set grad attribute to petrubation or zeroes depending on step_back
                    if self.set_grad:
                        params, grads = get_group_params_and_grads_tensorlist(group, with_grad=False, foreach=self.foreach)
                        # undo the petrubation
                        params.sub_(petrubation, alpha=group['magn'])
                        # accumulate gradients
                        if group['step_back']: grads.add_(petrubation)
                        # else gradients are 0

                    # or undo the petrubation and step back if step_back
                    else:
                        params = get_group_params_tensorlist(group, with_grad=False, foreach=self.foreach)
                        # if step back, go opposite way (magn undoes the petrubation and lr does the step)
                        if group['step_back']: params.sub_(petrubation, alpha=group['magn'] + group['lr'])
                        # else just undo the step
                        else: params.sub_(petrubation, alpha=group['magn'])

        if self.opt is not None: self.opt.step()
        return loss # type:ignore