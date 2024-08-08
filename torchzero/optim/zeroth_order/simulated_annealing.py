from collections.abc import Callable
from typing import Optional
import random, math
import torch
from torch import optim
from ..utils import get_group_params_tensorlist, get_group_params_and_grads_tensorlist
from .. import _foreach

__all__ = [
    "SimulatedAnnealing",
    "ThresholdAccepting",
]
class SimulatedAnnealing(optim.Optimizer):
    def __init__(
        self,
        params,
        lr:float = 1e-3,
        init_temp = 0.1,
        temp_mul = 0.9,
        iter_per_cool = 500,
        max_bad_iter = 50,
        sampler = torch.randn,
        stochastic = True,
        steps_per_sample = 1,
        foreach = True,
        log_temperature = False,
    ):
        defaults = dict(lr=lr, sampler=sampler)
        super().__init__(params, defaults)

        self.foreach = foreach
        self.stochastic = stochastic
        self.steps_per_sample = steps_per_sample
        self.temperature  = init_temp
        self.max_temperature = init_temp

        self.temp_mul = temp_mul
        self.iter_per_cool = iter_per_cool
        self.max_bad_iter = max_bad_iter

        self.current_step = 0
        self.lowest_loss = float('inf')
        self.global_lowest_loss = float('inf')
        self.bad_streak = 0

        self.log_temperature = log_temperature
        self.temperature_history = []

    @torch.no_grad
    def step(self, closure:Callable): # pylint:disable=W0222 # type:ignore
        # if stochastic, evaluate  loss on each step
        if self.stochastic or self.current_step == 0: self.lowest_loss = closure()

        groups_params = [(g, get_group_params_tensorlist(g, with_grad=False, foreach=self.foreach)) for g in self.param_groups]

        # we can do multiple steps per sample, only evaluating once per step
        for step in range(self.steps_per_sample):

            # create petrubations and apply them
            petrubations_per_group: list[_foreach.TensorList] = []
            for group, params in groups_params:
                group_petrubation = params.fn_like(group['sampler']).mul(group['lr'])
                petrubations_per_group.append(group_petrubation)
                params.add_(group_petrubation)

            # evaluate new loss
            loss = closure()

            if loss < self.global_lowest_loss:
                self.global_lowest_loss = loss
                self.bad_streak = 0

            # loss is lower or chance to go to a higher loss
            if loss < self.lowest_loss or random.random() < math.exp((self.lowest_loss - loss) / self.temperature):
                self.lowest_loss = loss

            else:
                for (group, params), petrubation in zip(groups_params, petrubations_per_group):
                    params.sub_(petrubation)

        self.current_step += 1
        if self.current_step % self.iter_per_cool == 0: self.temperature *= self.temp_mul
        if self.bad_streak > self.max_bad_iter:
            self.temperature *= self.temp_mul
            self.bad_streak = 0
        else: self.bad_streak += 1

        if self.log_temperature: self.temperature_history.append(self.temperature)
        return loss # type:ignore



class ThresholdAccepting(optim.Optimizer):
    def __init__(
        self,
        params,
        lr:float = 1e-3,
        init_threshold = 0.1,
        threshold_mul = 0.9,
        iter_per_decay = 500,
        max_bad_iter = 50,
        sampler = torch.randn,
        stochastic = True,
        steps_per_sample = 1,
        foreach = True,
        log_threshold = False,
    ):

        defaults = dict(lr=lr, sampler=sampler)
        super().__init__(params, defaults)

        self.foreach = foreach
        self.stochastic = stochastic
        self.steps_per_sample = steps_per_sample
        self.threshold  = init_threshold
        self.max_threshold = init_threshold

        self.threshold_mul = threshold_mul
        self.iter_per_decay = iter_per_decay
        self.max_bad_iter = max_bad_iter

        self.current_step = 0
        self.lowest_loss = float('inf')
        self.global_lowest_loss = float('inf')
        self.bad_streak = 0

        self.log_threshold = log_threshold
        self.threshold_history = []

    @torch.no_grad
    def step(self, closure:Callable): # pylint:disable=W0222 # type:ignore
        # if stochastic, evaluate  loss on each step
        if self.stochastic or self.current_step == 0: self.lowest_loss = closure()

        groups_params = [(g, get_group_params_tensorlist(g, with_grad=False, foreach=self.foreach)) for g in self.param_groups]

        # we can do multiple steps per sample, only evaluating once per step
        for step in range(self.steps_per_sample):

            # create petrubations and apply them
            petrubations_per_group: list[_foreach.TensorList] = []
            for group, params in groups_params:
                group_petrubation = params.fn_like(group['sampler']).mul(group['lr'])
                petrubations_per_group.append(group_petrubation)
                params.add_(group_petrubation)

            # evaluate new loss
            loss = closure()

            if loss < self.global_lowest_loss:
                self.global_lowest_loss = loss
                self.bad_streak = 0

            # loss is lower or chance to go to a higher loss
            if loss < self.lowest_loss + self.threshold:
                self.lowest_loss = loss

            else:
                for (group, params), petrubation in zip(groups_params, petrubations_per_group):
                    params.sub_(petrubation)

        self.current_step += 1
        if self.current_step % self.iter_per_decay == 0: self.threshold *= self.threshold_mul
        if self.bad_streak > self.max_bad_iter:
            self.threshold *= self.threshold_mul
            self.bad_streak = 0
        else: self.bad_streak += 1

        if self.log_threshold: self.threshold_history.append(self.threshold)
        return loss # type:ignore

