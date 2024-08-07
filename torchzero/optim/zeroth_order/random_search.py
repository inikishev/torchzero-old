from collections.abc import Callable, Sequence
from typing import Optional
import torch
from torch import optim
from ..utils import get_group_params_tensorlist, get_group_params_and_grads_tensorlist
from .. import _foreach
from ...random.random import Uniform, uniform

__all__ = [
    'RandomSearch',
    'RandomAnnealing',
    'AcceleratedRandomSearch',
]

class RandomSearch(optim.Optimizer):
    def __init__(
        self,
        params,
        sampler = Uniform(-1, 1),
        stochastic = False,
        steps_per_sample = 1,
        foreach=True,
        ):

        defaults = dict(sampler=sampler)
        super().__init__(params, defaults)
        self.foreach = foreach
        self.steps_per_sample=steps_per_sample
        self.stochastic=stochastic

        self.lowest_loss = float('inf')

        self.current_step = 0

    @torch.no_grad
    def step(self, closure:Callable): # type:ignore # pylint:disable = W0222
        if self.stochastic or self.current_step == 0: self.lowest_loss = closure()

        for step in range(self.steps_per_sample):
            prev = []
            for group in self.param_groups:
                params = get_group_params_tensorlist(group, with_grad=False, foreach=self.foreach)
                prev.append(params.clone())
                params.set_(params.fn_like(group['sampler']))

            loss = closure()

            if loss < self.lowest_loss:
                self.lowest_loss = loss

            else:
                for group, prev_params in zip(self.param_groups, prev):
                    params = get_group_params_tensorlist(group, with_grad=False, foreach=self.foreach)
                    params.set_(prev_params)

        self.current_step += 1
        return self.lowest_loss

class RandomAnnealing(optim.Optimizer):
    def __init__(
        self,
        params,
        n_steps: Optional[int],
        lr: float = 1,
        sampler = torch.randn,
        stochastic=False,
        steps_per_sample = 1,
        foreach=True,
    ):
        defaults = dict(lr=lr, sampler=sampler, n_steps = n_steps)
        super().__init__(params, defaults)
        self.foreach = foreach
        self.stochastic=stochastic
        self.steps_per_sample = steps_per_sample

        self.lowest_loss = float('inf')
        self.current_step = 0

    @torch.no_grad
    def step(self, closure:Callable): # type:ignore # pylint:disable = W0222
        if self.stochastic or self.current_step == 0: self.lowest_loss = closure()

        for step in range(self.steps_per_sample):

            prev = []
            for group in self.param_groups:
                params = get_group_params_tensorlist(group, with_grad=False, foreach=self.foreach)
                prev.append(params.clone())
                params.add_(params.fn_like(group['sampler']).mul(group['lr']))

            loss = closure()

            if loss < self.lowest_loss:
                self.lowest_loss = loss

            else:
                for group, prev_params in zip(self.param_groups, prev):
                    params = get_group_params_tensorlist(group, with_grad=False, foreach=self.foreach)
                    params.set_(prev_params)

        for group in self.param_groups:
            if group['n_steps'] is not None: group['lr'] -= group['lr']/group['n_steps']

        self.current_step += 1
        return self.lowest_loss


def _uniform_around_in_domain(point:_foreach.TensorList, around: float, domain:tuple[float, float], sampler=uniform):
    low, high = domain
    if high - low < 2 * around:
        raise ValueError(f"Domain is too small for `around`, {around = }, {domain = }")

    min_overflow = (low - (point.sub(around))).clamp_min(0)
    max_overflow = (high - (point.add(around))).clamp_max(0)

    point.add_(point.fn_like(sampler, -around, around).add(min_overflow).add(max_overflow))


class AcceleratedRandomSearch(optim.Optimizer):
    def __init__(
        self,
        params,
        domain: tuple[float, float] | Sequence,
        shrink = 0.9,
        restart_steps = 100,
        base_shrink = 1,
        base_shrink_mul = 1,
        sampler = uniform,
        stochastic = False,
        steps_per_sample = 1,
        foreach=True,
        ):
        """Accelerated random search (https://epubs.siam.org/doi/10.1137/S105262340240063X).

        Searches within a shrinking neigbourhood of the best parameters.
        When better parameters are found, search neighbourhood is reinitialized to entire search space.
        Search neighbourhood is also reinitialized when no better parameters are found for `restart_steps`.

        Args:
            params (_type_): Parameters to optimize.
            domain (tuple[float, float] | Sequence): Search domain, a `(low, high)` tuple.
            shrink (float, optional): Domain width will be multiplied by this after each step. Defaults to 0.9.
            restart_steps (int, optional): Domain will be reinitialized to entire search space times `base_shrink` after this many consecutive steps of not finding better parameters. Defaults to 100.
            base_shrink (float, optional): Width multiplier to reinitialize to.
            base_shrink_mul (float, optional): Multiply base_shrink by this each time better parameters are found.
            sampler (_type_, optional): Random sampler, a callable that accepts (shape, low, high) args. Defaults to uniform.
            stochastic (bool, optional): If True, evaluates closure before and after each step. Defaults to False.
            steps_per_sample (int, optional): Steps to do per each sample (only useful with stochastic = True). Defaults to 1.
            foreach (bool, optional): Makes it faster. Defaults to True.

        The following parameters support parameter groups:
        - domain
        - shrink
        - base_shrink
        - base_shrink_mul
        - sampler
        - restart_steps
        """
        defaults = dict(
            sampler=sampler,
            domain=domain,
            shrink=shrink,
            base_shrink=base_shrink,
            base_shrink_mul=base_shrink_mul,
            current_shrink=base_shrink,
            restart_steps=restart_steps,
            current_steps = 0,
        )
        super().__init__(params, defaults)
        self.foreach = foreach
        self.steps_per_sample=steps_per_sample
        self.stochastic=stochastic

        self.lowest_loss = float('inf')

        self.current_step = 0

    @torch.no_grad
    def step(self, closure:Callable): # type:ignore # pylint:disable = W0222
        if self.stochastic or self.current_step == 0: self.lowest_loss = closure()

        for step in range(self.steps_per_sample):
            prev = []
            for group in self.param_groups:
                params = get_group_params_tensorlist(group, with_grad=False, foreach=self.foreach)
                prev.append(params.clone())
                # generate params within shrinking neighborhoods of a previous record-generating value
                domain = group['domain']
                around = ((domain[1] - domain[0]) / 2) * group['current_shrink']
                _uniform_around_in_domain(params, around = around, domain = domain, sampler = group['sampler'])

            loss = closure()

            if loss < self.lowest_loss:
                self.lowest_loss = loss
                for group in self.param_groups:
                    # reinitialize search neighborhood when a new record is found
                    group['current_shrink'] = group['base_shrink']
                    group['current_steps'] = 0

            else:
                for group, prev_params in zip(self.param_groups, prev):
                    params = get_group_params_tensorlist(group, with_grad=False, foreach=self.foreach)
                    params.set_(prev_params)
                    # shrink search neighborhood
                    group['current_shrink'] *= group['shrink']
                    group['current_steps'] += 1
                    # reinitialize the search neighborhood after some number of unsuccessful shrink steps
                    if group['current_steps'] > group['restart_steps']:
                        group['current_steps'] = 0
                        group['current_shrink'] = group['base_shrink']
                        group['base_shrink'] *= group['base_shrink_mul']

        self.current_step += 1
        return self.lowest_loss
