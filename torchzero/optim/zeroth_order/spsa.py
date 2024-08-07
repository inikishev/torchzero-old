from collections.abc import Callable
from typing import Optional, Literal
import logging
import torch
from torch import optim
from ..utils import get_group_params_tensorlist, get_group_params_and_grads_tensorlist
from .. import _foreach
from ...random.random import rademacher

__all__ = [
    "SPSA",
]
class SPSA(optim.Optimizer):
    def __init__(
        self,
        params,
        lr:Optional[float] = 1e-3,
        magn:float = 1e-5,
        max_diff: Optional[float] = 1e-2,
        min_diff: Optional[float] = 1e-8,
        sampler:Callable = rademacher,
        variant:Literal['SPSA', "RDSA", "2step"] = 'SPSA',
        set_grad = False,
        opt = None,
        avg_steps = 1,
        foreach = True,
        verbose=False,
    ):
        """Simultaneous perturbation stochastic approximation (SPSA), random direction stochastic approximation (RDSA), and two-step random search.

        Spall, James C. "A stochastic approximation technique for generating maximum likelihood parameter estimates." 1987 American control conference. IEEE, 1987.

        Args:
            params (Iterable[torch.Tensor]): Iterable of parameters to optimize. Usually `model.parameters()`.
            lr (float | None): Learning rate. This has no effect if `set_grad` is True.
            magn (float | None): Random direction magnitude.
            max_diff (float | None): Limits `loss_poss - loss_neg` by this value, to improve stability by avoiding large steps If you experience loss going up, try decreasing this, potentially up to 1e-6.
            sampler (Callable like torch.randn): Random petrubation sampler. SPSA has to use a distribution with only 1s and -1s, like rademacher, which is the default. RDSA and 2-step variants can use any zero-mean distribution.
            set_grad (bool): If `True`, this acts like a gradient approximator and sets `grad` attribute, otherwise this acts as a standalone optimizer and doesn't touch `grad` attribute. Defaults to False.
            opt (torch.optim.Optimizer | any object with `step` method): Do a step with any gradient-based optimizer after approximating gradients. Defaults to None.
            avg_steps (int): Do multiple approximations and take their average. Defaults to 1.
            foreach (bool): Use `foreach` operations which makes it much faster. Defaults to True.
            fast_sampling (bool): Uses about 50% faster random sampling, but this generates one petrubation and reuses it for all parameters. That doesn't matter for when parameters are big, i.e. for neural networks, but if you have parameters with a single value, they will be synchronized. Defaults to False.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """
        if lr is None and magn is None: raise ValueError("Either lr or magn must be specified.")
        if magn is None: magn = lr
        if set_grad is False and opt is not None: raise ValueError("opt can only be used with set_grad=True")
        if lr is None and not set_grad: raise ValueError("lr must be specified if set_grad=False")

        defaults = dict(lr=lr, magn=magn, sampler=sampler, variant=variant, min_diff = min_diff, max_diff=max_diff)
        super().__init__(params, defaults)

        self.foreach = foreach
        self.set_grad = set_grad
        self.opt = opt
        self.avg_steps = avg_steps

        self.verbose = verbose

    @torch.no_grad
    def step(self, closure:Callable): # pylint:disable=W0222 # type:ignore

        grads_per_group_per_step:list[list[_foreach.TensorList]] = []

        for step in range(self.avg_steps):
            # create petrubations and add them
            petrubations_per_group: list[_foreach.TensorList] = []
            for group in self.param_groups:
                params = get_group_params_tensorlist(group, with_grad=False, foreach=self.foreach)
                group_petrubation = params.fn_like(group['sampler'])
                petrubations_per_group.append(group_petrubation)
                if step == 1: params.add_(group_petrubation, alpha = group['magn'])
                else: params.add_(group_petrubation, alpha = group['magn'] * 2)

            # evaluate positive loss
            loss_pos = closure()

            # subtract petrubations
            for group, petrubation in zip(self.param_groups, petrubations_per_group):
                params = get_group_params_tensorlist(group, with_grad=False, foreach=self.foreach)
                params.sub_(petrubation, alpha = 2 * group['magn'])

            # evaluate negative loss
            loss_neg = closure()

            # apply SPSA formula and add results
            this_step_grads_per_group = []
            for group, petrubation in zip(self.param_groups, petrubations_per_group):

                variant = group['variant']
                max_diff = group['max_diff']

                # limit dloss
                dloss = loss_pos - loss_neg
                if max_diff is not None and abs(dloss) > max_diff: dloss = max_diff * (1 if dloss > 0 else -1)

                if self.verbose: print(f'{loss_pos = }, {loss_neg = }, {dloss = }')

                # apply the formula
                if variant == 'SPSA':
                    if self.verbose: print(f'{petrubation.unique(False)}')
                    if self.verbose: print(f'{petrubation.pow(-1).max() = }')
                    this_step_grads_per_group.append(petrubation.pow(-1.).mul(dloss / (2 * group['magn'])))

                elif variant == 'RDSA':
                    this_step_grads_per_group.append(petrubation.mul(dloss / (2 * group['magn'])))

                elif variant == '2step':
                    if loss_pos > loss_neg: this_step_grads_per_group.append(petrubation)
                    else: this_step_grads_per_group.append(petrubation.mul(-1))

                else: raise ValueError(f"Invalid variant {group['variant']}")

            grads_per_group_per_step.append(this_step_grads_per_group)

        if self.avg_steps == 1: averaged_grads_per_group = grads_per_group_per_step[0]
        else: averaged_grads_per_group = [_foreach.sequencemean(i, foreach=self.foreach) for i in zip(*grads_per_group_per_step)]

        # apply SPSA update
        for group, spsa_grad, last_petr in zip(self.param_groups, averaged_grads_per_group, petrubations_per_group): # type:ignore
            if self.set_grad:
                params, grads = get_group_params_and_grads_tensorlist(group, with_grad=False, foreach=self.foreach)
                # undo the step
                params.add_(last_petr, alpha=group['magn'])
                # accumulate gradients
                grads.add_(spsa_grad)
            else:
                # get only params
                params = get_group_params_tensorlist(group, with_grad=False, foreach=self.foreach)
                # undo the step
                params.add_(last_petr, alpha=group['magn'])
                # apply the update
                params.sub_(spsa_grad, alpha=group['lr'])

        if self.opt is not None: self.opt.step()
        return min(loss_pos, loss_neg) # type:ignore
