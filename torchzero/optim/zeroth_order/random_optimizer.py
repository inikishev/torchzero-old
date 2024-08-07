from collections.abc import Callable
from typing import Optional, cast
from itertools import count
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
        lr:float = 1e-3,
        step_back = False,
        sampler = torch.randn,
        best_of = 1,
        set_grad = False,
        opt = None,
        stochastic = True,
        steps_per_sample = 1,
        foreach = True,
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
        if set_grad is False and opt is not None: raise ValueError("opt can only be used with set_grad=True")

        defaults = dict(lr=lr, step_back=step_back, sampler=sampler)
        super().__init__(params, defaults)

        self.foreach = foreach
        self.set_grad = set_grad
        self.opt = opt
        self.stochastic = stochastic

        if best_of < 1: raise ValueError(f"best_of must be at least 1, got {best_of}")
        self.best_of = best_of

        if steps_per_sample < 1: raise ValueError(f"steps_per_sample must be at least 1, got {steps_per_sample}")
        # step back means model steps in the opposite way, where loss might be higher, so lowest_loss would no longer be relevant
        if step_back and not stochastic: raise ValueError("step_back can only be used with stochastic=True")
        self.steps_per_sample = steps_per_sample

        self.current_step = 0
        self.lowest_loss = float('inf')

    @torch.no_grad
    def step(self, closure:Callable): # pylint:disable=W0222 # type:ignore
        # if stochastic, evaluate  loss on each step
        if self.stochastic or self.current_step == 0: self.lowest_loss = closure()

        groups_params = [(g, get_group_params_tensorlist(g, with_grad=False, foreach=self.foreach)) for g in self.param_groups]

        for _ in range(self.steps_per_sample):

            petrubations_per_group = cast(list[_foreach.TensorList], None)
            best_petrubations_per_group = cast(list[_foreach.TensorList], None) # make it iterable
            loss_improved = False

            for bestof_iter in range(self.best_of):
                petrubations_per_group: list[_foreach.TensorList] = []
                iter_is_best = False
                for group, params in groups_params:
                    # generate petrubations
                    petrubation = params.fn_like(group['sampler']).mul(group['lr'])
                    petrubations_per_group.append(petrubation)
                    # apply it
                    params.add_(petrubation)

                # evaluate new loss
                loss = closure()

                # if loss improved
                if loss < self.lowest_loss:
                    self.lowest_loss = loss
                    best_petrubations_per_group = petrubations_per_group
                    loss_improved = True
                    iter_is_best = True

                if best_petrubations_per_group is None: best_petrubations_per_group = petrubations_per_group
                for (group, params), last_petrubation, best_petrubation in zip(groups_params, petrubations_per_group, best_petrubations_per_group):
                    # restore params if not last iteration
                    if bestof_iter != self.best_of - 1: params.sub_(last_petrubation)
                    # after last iteration apply best petrubation
                    else:
                        # use as gradient approx
                        if self.set_grad:
                            # we do this again because it creates grad
                            _, grads = get_group_params_and_grads_tensorlist(group, with_grad=False, foreach=self.foreach)
                            # restore params
                            params.sub_(last_petrubation)
                            # accumulate gradients
                            if loss_improved: grads.sub_(best_petrubation)
                            elif group['step_back']: grads.add_(best_petrubation)
                        # use as optimizer
                        else:
                            # if one of the petrubations is better
                            # print(f'{self.lowest_loss = }, {loss = }, {loss_improved = }, {iter_is_best = }, {last_petrubation = }, {best_petrubation = }')
                            if loss_improved:
                                # if the best petrubation isn't the last one that is already applied
                                if not iter_is_best:
                                    # undo last petrubation and apply best one
                                    params.sub_(last_petrubation)
                                    params.add_(best_petrubation)
                            # no petrubations are better, if step back, go opposite way
                            elif group['step_back']:
                                # we go opposite of last petrubation, which is typically the only one anyway
                                params.sub_(last_petrubation.mul(2))
                            else:
                                # we undo petrubation
                                params.sub_(last_petrubation)


        if self.opt is not None: self.opt.step()
        self.current_step += 1
        return loss # type:ignore