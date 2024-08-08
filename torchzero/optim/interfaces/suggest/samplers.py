from typing import Optional, Any, Literal
from collections.abc import Callable

import torch

from ....optim import (
    RandomSearch,
    AcceleratedRandomSearch,
    GridSearch,
    SequentialSearch,
    SimulatedAnnealing,
    ThresholdAccepting,
    SPSA,
    FDSA,
    RandomOptimizer,
    RandomAnnealing,
)
from ....random.random import Uniform, uniform, rademacher
from .study import Study

def _get_domain(mod, default_domain: Optional[tuple[float, float]]):
    """mod must be a parameter module, if it has `true_min` and `true_max` defined, they will be usd

    :param mod: ParamNumeric or a child.
    :param default_domain: (min_value, mav_value)
    :return: (unscaled_min, unscaled_max)
    """
    if mod.unscaled_min is None:
        if default_domain is None: raise ValueError('default_domain must be provided if true_min is None')
        min = default_domain[0]
    else: min = mod.unscaled_min
    if mod.unscaled_max is None:
        if default_domain is None: raise ValueError('default_domain must be provided if true_max is None')
        max = default_domain[1]
    else: max = mod.unscaled_max
    return min, max

class RandomSearchSampler(RandomSearch):
    def __init__(self, study:Study, default_domain: Optional[tuple[float, float]] = None, foreach=True):
        params = []
        for mod in study.trial.children():
            options = {'params': mod.parameters(), "sampler": Uniform(*_get_domain(mod, default_domain))}
            options.update(mod.options)
            params.append(options)

        super().__init__(params, stochastic=False, steps_per_sample=1, foreach=foreach)


class RandomAnnealingSampler(RandomAnnealing):
    def __init__(self, study:Study, n_steps, lr = 1., sampler = torch.randn, foreach=True):
        params = []
        for mod in study.trial.children():
            options = {'params': mod.parameters()}
            options.update(mod.options)
            params.append(options)

        super().__init__(params, stochastic=False, sampler = sampler, n_steps=n_steps, lr= lr, foreach=foreach)
        
class AcceleratedRandomSearchSampler(AcceleratedRandomSearch):
    def __init__(self,
                study:Study,
                shrink: float = 0.9,
                restart_steps: int = 100,
                base_shrink: int = 1,
                base_shrink_mul: int = 1,
                default_domain: Optional[tuple[float, float]] = None,
                sampler: Callable = uniform,
                foreach: bool = True,
                ):
        params = []
        for mod in study.trial.children():
            options = {'params': mod.parameters(), "domain": _get_domain(mod, default_domain)}
            options.update(mod.options)
            params.append(options)

        super().__init__(
            params,
            domain = default_domain if default_domain is not None else (-1, 1),
            shrink=shrink,
            restart_steps=restart_steps,
            base_shrink=base_shrink,
            base_shrink_mul=base_shrink_mul,
            sampler=sampler,
            steps_per_sample=1,
            stochastic=False,
            foreach=foreach
        )


class GridSearchSampler(GridSearch):
    def __init__(self, study:Study, default_domain: Optional[tuple[float, float]] = None, step:Optional[float] = None):
        params = []
        for mod in study.trial.children():
            # clipping params will break grid search
            if mod.clip_params:
                mod.clip_params = False # type:ignore
                mod.clip_value = True # type:ignore

            options = {'params': mod.parameters(), "domain": _get_domain(mod, default_domain)}
            options.update(mod.options)
            params.append(options)

        super().__init__(
            params,
            domain = default_domain if default_domain is not None else (-1, 1),
            step = step,
            )

class SequentialSearchSampler(SequentialSearch):
    def __init__(self, study:Study, step:float, step_mul = 1., default_domain: Optional[tuple[float, float]] = None):
        params = []
        for mod in study.trial.children():
            # clipping params will break sequential search
            if mod.clip_params:
                mod.clip_params = False # type:ignore
                mod.clip_value = True # type:ignore

            options = {'params': mod.parameters(), "domain": _get_domain(mod, default_domain)}
            options.update(mod.options)
            params.append(options)

        super().__init__(
            params,
            step = step,
            step_mul = step_mul,
            domain = default_domain if default_domain is not None else (-1, 1),
            )

class SimulatedAnnealingSampler(SimulatedAnnealing):
    def __init__(self, study:Study,
                lr: float = 0.001,
                init_temp: float = 0.1,
                temp_mul: float = 0.9,
                iter_per_cool: int = 500,
                max_bad_iter: int = 50,
                sampler = torch.randn,
                foreach: bool = True,
                log_temperature: bool = False
                ):
        params = []
        for mod in study.trial.children():
            options: dict[str, Any] = {'params': mod.parameters()}
            options.update(mod.options)
            params.append(options)

        super().__init__(params,
                         lr = lr,
                         init_temp = init_temp,
                         temp_mul = temp_mul,
                         iter_per_cool = iter_per_cool,
                         max_bad_iter = max_bad_iter,
                         sampler = sampler,
                         foreach = foreach,
                         log_temperature = log_temperature,
                         stochastic = False,
                         steps_per_sample = 1,
                         )


class ThresholdAcceptingSampler(ThresholdAccepting):
    def __init__(self, study:Study,
                lr: float = 0.001,
                init_threshold: float = 0.1,
                threshold_mul: float = 0.9,
                iter_per_decay: int = 500,
                max_bad_iter: int = 50,
                sampler = torch.randn,
                foreach: bool = True,
                log_threshold: bool = False
                ):

        params = []
        for mod in study.trial.children():
            options: dict[str, Any] = {'params': mod.parameters()}
            options.update(mod.options)
            params.append(options)

        super().__init__(params,
                         lr = lr,
                         init_threshold = init_threshold,
                         threshold_mul = threshold_mul,
                         iter_per_decay = iter_per_decay,
                         max_bad_iter = max_bad_iter,
                         sampler = sampler,
                         foreach = foreach,
                         log_threshold = log_threshold,
                         stochastic = False,
                         steps_per_sample = 1,
                         )

class SPSASampler(SPSA):
    def __init__(self, study:Study,
                lr: float | None = 1e-3,
                magn: float = 1e-4,
                max_diff: float | None = 1e-2,
                min_diff: float | None = 1e-8,
                sampler = rademacher,
                variant: Literal['SPSA', 'RDSA', '2step'] = 'SPSA',
                set_grad: bool = False,
                opt: torch.optim.Optimizer | Any | None = None,
                avg_steps: int = 1,
                foreach: bool = True,
                verbose: bool = False
                ):

        params = []
        for mod in study.trial.children():
            options: dict[str, Any] = {'params': mod.parameters()}
            options.update(mod.options)
            params.append(options)

        super().__init__(params,
                         lr = lr,
                         magn=magn,
                         max_diff=max_diff,
                         min_diff=min_diff,
                         sampler=sampler,
                         variant=variant,
                         set_grad=set_grad,
                         opt=opt,
                         avg_steps=avg_steps,
                         foreach=foreach,
                         verbose=verbose,
                         )

class FDSASampler(FDSA):
    def __init__(self, study:Study,
                lr: float | None = 0.001,
                magn: float = 0.001,
                set_grad: bool = False,
                opt: Any | None = None
                ):

        params = []
        for mod in study.trial.children():
            options: dict[str, Any] = {'params': mod.parameters()}
            options.update(mod.options)
            params.append(options)

        super().__init__(params,
                        lr = lr,
                        magn = magn,
                        set_grad = set_grad,
                        opt = opt)


class RandomOptimizerSampler(RandomOptimizer):
    def __init__(self, study:Study,
                lr: float = 1e-3,
                best_of: int = 1,
                sampler = torch.randn,
                set_grad: bool = False,
                opt: torch.optim.Optimizer | Any | None = None,
                foreach: bool = True,
                ):

        params = []
        for mod in study.trial.children():
            options: dict[str, Any] = {'params': mod.parameters()}
            options.update(mod.options)
            params.append(options)

        super().__init__(params,
                         lr = lr,
                         step_back=False,
                         best_of=best_of,
                         sampler=sampler,
                         set_grad=set_grad,
                         opt=opt,
                         foreach=foreach,
                         stochastic=False,
                         )


