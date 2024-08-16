"""Wraps optimizers from https://github.com/ltecot/rand_bench_opt_quantum"""
from collections.abc import Callable
import torch
from torch.optim import Optimizer

def get_param_len(params):
    return torch.nn.utils.parameters_to_vector(params).numel()

class RBOQWrapper(Optimizer):
    def __init__(self, params, optimizer):
        """This wraps optimizers from https://github.com/ltecot/rand_bench_opt_quantum, I HAVE NOT TESTED IF IT WORKS YET."""
        params = list(params)
        self.params = params
        self.params_vec = torch.nn.utils.parameters_to_vector(params)
        self.param_len = self.params_vec.numel()
        self.optimizer=optimizer

        super().__init__(params, {})

    def step(self, closure: Callable):  # type:ignore #pylint:disable=W0222
        def objective_fn(new_param_vec):
            backup = torch.nn.utils.parameters_to_vector(self.params).clone()
            torch.nn.utils.vector_to_parameters(new_param_vec, self.params)
            loss = closure()
            torch.nn.utils.vector_to_parameters(backup, self.params)
            return loss

        params, loss = self.optimizer.step_and_cost(objective_fn, self.params_vec)
        torch.nn.utils.vector_to_parameters(params, self.params)
        return loss