from collections.abc import Callable
import math
import torch
from torch.optim import Optimizer

from .utils import foreach_param, foreach_group_param
from ._utils_gridsearch import _map_to_base,_extend_to_length, _GridSearch, iproduct

def gridsearch_vec_combinations(num, step, domain):
    """Yields all possible parameter combinations for a vector of length `num` with given step size in given domain"""
    domain_normalized = [int(i/step) for i in domain] # we get 3, 4, 5
    add_term = domain_normalized[0]
    domain_from0 = [i - add_term for i in domain_normalized] # we get 0, 1, 2
    base = domain_from0[1] + 1

    num_combinations = base ** num
    for c in range(num_combinations):
        yield (_extend_to_length(_map_to_base(c, base), num) + add_term) * step

def gridsearch_param_combinations(param_nums, steps, domains):
    """Yields all possible parameter combinations for a list of vectors of lengths given in `param_nums`,
    with steps and domains for each vector given in `steps` and `domains`"""
    params = zip(param_nums,steps,domains)
    gridsearches = [_GridSearch(p,s,d) for p,s,d in params]
    combination_idxs = iproduct(*[range(gs.num_combinations) for gs in gridsearches])
    for combination in combination_idxs:
        yield [gs.get(c) for gs, c in zip(gridsearches, combination)]

class GridSearch(Optimizer):
    """Grid search. Tries every combination of parameters possible. After trying all combinations this will halve the step size and restart."""
    def __init__(self, params, domain: tuple[float, float], step:float):
        defaults = dict(domain=domain, step=step)
        super().__init__(params, defaults)

        self.lowest_loss = float("inf")
        self.n_steps = 0

        domains = []
        steps = []
        for group in self.param_groups:
            domains.append(group["domain"])
            steps.append(group["step"])

        self._param_combinations_generator = gridsearch_param_combinations([p.numel() for p in foreach_param(self.param_groups)], steps, domains)
        self._sizes = [p.size() for p in foreach_param(self.param_groups)]


    @torch.no_grad
    def step(self, closure:Callable): # type:ignore #pylint:disable=W0222
        # on first iteration we calculate the initial loss to compare new parameters to on next iterations
        if self.n_steps == 0:
            # save all parameters
            for p in foreach_param(self.param_groups):
                state = self.state[p]
                state['best'] = p.clone()

            # evaluate the initial loss
            self.lowest_loss = closure()

        # make a step
        try: flat_params = next(self._param_combinations_generator)
        except StopIteration:
            domains = []
            steps = []
            for group in self.param_groups:
                domains.append(group["domain"])
                group["step"] /= 2
                steps.append(group["step"])
                self._param_combinations_generator = gridsearch_param_combinations([p.numel() for p in foreach_param(self.param_groups)], steps, domains)
            return self.lowest_loss

        # copy gridsearch params
        for i,p in enumerate(foreach_param(self.param_groups)):
            p.copy_(flat_params[i].view(self._sizes[i]))

        # test new params
        loss = closure()
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            for p in foreach_param(self.param_groups):
                state = self.state[p]
                state['best'] = p.clone()

        else:
            for p in foreach_param(self.param_groups): p.copy_(self.state[p]['best'])

        return loss

class SequentialSearch(Optimizer):
    """Sequential search. Finds the best value of first parameter, then second parameter, etc. After fitting all parameters the process is restarted from the new position."""
    def __init__(self, params, lr, domain: tuple[float, float]):
        defaults = dict(domain=domain, lr=lr)
        super().__init__(params, defaults)

        self.lowest_loss = float("inf")
        self.n_steps = 0

        self._group_param_flat = [(g, p, p.ravel()) for g, p in list(foreach_group_param(self.param_groups))]

    @torch.no_grad
    def step(self, closure:Callable): # type:ignore #pylint:disable=W0222
        # on first iteration we calculate the initial loss to compare new parameters to on next iterations
        if self.n_steps == 0:
            # save all parameters
            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]
                state['best'] = p.clone()

                # get domain minimum
                group_min = group["domain"][0]
                state['cur_value'] = group_min
                # p.fill_(group_min)

            # evaluate the initial loss
            self.lowest_loss = closure()

            self.cur_param = 0
            self.cur_index = 0

        # make a step
        # if went through all parameters, restart
        if self.cur_param >= len(self._group_param_flat):
            self.cur_param = 0

        group, p, pflat = self._group_param_flat[self.cur_param]
        # if current index is within current parameter
        if self.cur_index < len(pflat):
            state = self.state[p]

            # if current value is within the domain
            value = state['cur_value']
            if value <= group['domain'][1]:
                pflat[self.cur_index] = value
                state['cur_value'] += group["lr"]
            # else advance the index and reset current value
            else:
                self.cur_index += 1
                state['cur_value'] = group['domain'][0]
        # else move to next parameter and reset the index
        else:
            self.cur_param += 1
            self.cur_index = 0

        loss = closure()
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            for p in foreach_param(self.param_groups): self.state[p]['best'] = p.clone()

        else:
            for p in foreach_param(self.param_groups): p.copy_(self.state[p]['best'])

        self.n_steps += 1
        return loss