from collections.abc import Callable, Sequence
import math
import torch
from torch.optim import Optimizer

from .utils import foreach_param, foreach_group_param
from ._utils_gridsearch import _map_to_base,_extend_to_length, _GridSearch, iproduct

def gridsearch_vec_combinations(num:int, step:float, domain:tuple[float,float]):
    """Yields all possible parameter combinations for a vector of length `num` with given step size in given domain.

    Args:
        num (int): length of parameter vector i.e. number of parameters.
        step (float): search step, e.g. at 0.1 it will increment each parameter by 0.1.
        domain (tuple[float,float]): search domain, e.g. `(-1, 1)` will search between -1 and 1.

    Yields:
        torch.Tensor: vector of parameters.
    """
    domain_normalized = [int(i/step) for i in domain] # we get 3, 4, 5
    add_term = domain_normalized[0]
    domain_from0 = [i - add_term for i in domain_normalized] # we get 0, 1, 2
    base = domain_from0[1] + 1

    num_combinations = base ** num
    for c in range(num_combinations):
        yield (_extend_to_length(_map_to_base(c, base), num) + add_term) * step

def gridsearch_param_combinations(param_nums: Sequence[int], steps: Sequence[float], domains:Sequence[tuple[float,float]]):
    """Yields all possible parameter combinations for a list of vectors of lengths given in `param_nums`,
    with steps and domains for each vector given in `steps` and `domains`.

    Args:
        param_nums (Sequence[int]): Sequence of lengths of each parameter vector i.e. numbers of parameters.
        steps (Sequence[float]): Sequence of search steps for each parameter, e.g. at 0.1 it will increment each parameter by 0.1.
        domains (Sequence[tuple[float,float]]): Sequence of search domains for each parameter, e.g. `(-1, 1)` will search between -1 and 1.

    Yields:
        list[torch.Tensor]: list of parameter vectors.
    """
    params = zip(param_nums,steps,domains)
    gridsearches = [_GridSearch(p,s,d) for p,s,d in params]
    combination_idxs = iproduct(*[range(gs.num_combinations) for gs in gridsearches])
    for combination in combination_idxs:
        yield [gs.get(c) for gs, c in zip(gridsearches, combination)]

class GridSearch(Optimizer):
    def __init__(self, params, domain: tuple[float, float], step:float):
        """Grid search.
        Tries every combination of parameters possible.
        After trying all combinations this will halve the step size and restart.

        This evaluates closure once per step.

        Grid search is typically used for small-dimensional problems like finding some small subset of optimal hyperparameters.

        The amount of steps needed to perform a full grid search can be calculated as `steps_in_domain ^ num_parameters`,
        where `steps_in_domain` is how many step-sizes fit into the search domain.
        If you have 10 parameters and domain of 10 step-sizes,
        you will already have to perform `10^10 = 10,000,000,000` steps for a full grid search.
        Thus this is only suitable for very small-dimensional problems.

        Args:
            params: Parameters to optimize, usually `model.parameters()`.

            domain (tuple[float, float]): search domain, e.g. `(-1, 1)` will search between -1 and 1. Supports parameter groups.

            step (float): search step, e.g. at 0.1 it will increment each parameter by 0.1. Supports parameter groups.
        """
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
        """Performs a single optimization step (parameter update).

        Args:
            closure (Callable): A closure that reevaluates the model and returns the loss. The closure is evaluated twice on the first step, and then once per step.

        """
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
        # After trying all combinations this halves the step size and restarts.
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
    def __init__(self, params, lr:float, domain: tuple[float, float]):
        """Sequential search. Sequentially finds the best value for each individual parameter,
        i.e. it tries all values of 1st parameter and picks the best, then tries all values of 2nd parameter and picks the best, and so on.
        After fitting all parameters the process is restarted from the best found position.

        This evaluates closure once per step.

        This may be faster than a grid search for certain problems where parameters don't depend on each other strongly.
        If you have 10 parameters and a domain of 10 step-sizes,
        one iteration will require `10 * 10 = 100` evaluations, which is way lower than grid search.
        However if parameters depend on each other, you may need many iterations to reach the minimum, or it might never converge,
        and it isn't as thorough as grid search.

        Args:
            params: Parameters to optimize, usually `model.parameters()`.

            lr (float): search step, e.g. at 0.1 it will increment each parameter by 0.1. Supports parameter groups and lr schedulers.

            domain (tuple[float, float]): search domain, e.g. `(-1, 1)` will search between -1 and 1. Supports parameter groups.
        """
        defaults = dict(domain=domain, lr=lr)
        super().__init__(params, defaults)

        self.lowest_loss = float("inf")
        self.n_steps = 0

        self._group_param_flat = [(g, p, p.ravel()) for g, p in list(foreach_group_param(self.param_groups))]

    @torch.no_grad
    def step(self, closure:Callable): # type:ignore #pylint:disable=W0222
        """Performs a single optimization step (parameter update).

        Args:
            closure (Callable): A closure that reevaluates the model and returns the loss. The closure is evaluated twice on the first step, and then once per step.

        """
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