from collections.abc import Callable
from typing import Optional
import random, math
import torch
from torch import optim
from ..utils import get_group_params_tensorlist, get_group_params_and_grads_tensorlist
from .. import _foreach

__all__ = [
    "SimulatedAnnealing",
]
class SimulatedAnnealing(optim.Optimizer):
    def __init__(
        self,
        params,
        cooling_steps: int,
        lr:float = 1e-3,
        init_temp = 0.1,
        sampler = torch.randn,
        stochastic = True,
        steps_per_sample = 1,
        foreach = True,
    ):
        """Moves to better parameters, and has a chance to move to worse parameters based on temperature, to escape local minima.

        Args:
            params (_type_): _description_
            cooling_steps (int): Temperature will linearly decay to 0 over this many steps.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            init_temp (float, optional): Initial temperature. Defaults to 0.1.
            sampler (_type_, optional): Random sampler. Defaults to torch.randn.
            stochastic (bool, optional): _description_. Defaults to True.
            steps_per_sample (int, optional): _description_. Defaults to 1.
            foreach (bool, optional): _description_. Defaults to True.
        """

        defaults = dict(lr=lr, sampler=sampler)
        super().__init__(params, defaults)

        self.foreach = foreach
        self.stochastic = stochastic
        self.steps_per_sample = steps_per_sample
        self.temperature  = init_temp
        self.max_temperature = init_temp

        self.cooling_steps = cooling_steps
        self.current_step = 0
        self.lowest_loss = float('inf')

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

            if loss < self.lowest_loss: probability = 1
            else: 
                probability = math.exp((self.lowest_loss - loss) / self.temperature)

            # loss is lower or chance to go to a higher loss
            if loss < self.lowest_loss or random.random() < probability:
                self.lowest_loss = loss

            else:
                # revert petrubations
                for (group, params), petrubation in zip(groups_params, petrubations_per_group):
                    params.sub_(petrubation)

        self.heat = self.max_temperature * (1 - self.current_step / self.cooling_steps)

        self.current_step += 1
        return loss # type:ignore