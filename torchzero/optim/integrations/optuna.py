from typing import Any
import torch
import numpy as np
import optuna

from ..zeroth_order.grid_search import _gridsearch_param_combinations

class GridSearch(optuna.samplers.BaseSampler):
    """Sampler using a more convenient form of grid search.

    Tries every combination of parameters possible.
    After trying all combinations this will halve the step size and restart, while making sure not to try same combination twice.
    As a result, you don't have to worry about setting `step` and appropriate number of iterations,
    since it will evenly explore the search space with more and more precision regardless of how many iterations you run it for.

    This supports int, float and categorical variables (not log yet).

    This is a optuna sampler version of `GridSearch` optimizer.

    """
    def __init__(self):
        super().__init__()

        self.params = {}

    def sample_relative(self, study, trial, search_space):
        if search_space == {}:
            return {}

        # add params to model on 1st step
        if len(self.params) == 0:
            for param_name, param_distribution in search_space.items():
                # float
                if isinstance(param_distribution, (optuna.distributions.IntDistribution, optuna.distributions.FloatDistribution)):
                    if not (param_distribution.step is None or param_distribution.step == 1.):
                        raise NotImplementedError("step is not supported (idk what that means)")
                    if param_distribution.log:
                        raise NotImplementedError('log is yet to be supported')
                    if param_name not in self.params:
                        p = self.params[param_name] = {}
                        p['value'] = param_distribution.low
                        p['domain'] = [param_distribution.low, param_distribution.high]
                        p['step'] = (param_distribution.high - param_distribution.low) / 2
                        p['add_term'] = 0
                        if isinstance(param_distribution, optuna.distributions.FloatDistribution) : p['type'] = 'float'
                        else: p['type'] = 'int'
                        if p['type'] == 'int': p['step'] = int(max(1, int(p['step'])))

                elif isinstance(param_distribution, optuna.distributions.CategoricalDistribution):
                    if param_name not in self.params:
                        p = self.params[param_name] = {}
                        p['value'] = 0
                        p['domain'] = [0, len(param_distribution.choices) - 1]
                        p['choices'] = param_distribution.choices
                        p['step'] = 1
                        p['add_term'] = 0
                        p['type'] = 'categorical'

                else: raise ValueError(f'Unsupported distribution: {param_distribution}')
            self._param_combinations_generator = _gridsearch_param_combinations([1 for i in self.params],
                                                                                [p['step'] for p in self.params.values()],
                                                                                [p['domain'] for p in self.params.values()])


        # make a step
        try: flat_params = next(self._param_combinations_generator)
        # After trying all combinations this halves the step size and restarts.
        except StopIteration:
            domains = []
            steps = []
            for p in self.params.values():
                # set step if None
                if p['step'] is None: p['step'] = (p["domain"][1] - p["domain"][0]) / 2
                # divide step by two for next set of iterations
                else: p["step"] /= 2

                if p['type'] == 'int': p['step'] = int(max(1, int(p['step'])))
                elif p['type'] == 'categorical': p['step'] = 1

                # we add step * 0.5 to parameters to make sure we don't try same parameters twice
                p['add_term'] = p['step'] / 2
                steps.append(p["step"])
                # since we add step * 0.5, we make domain that much smaller to make sure we don't get out of the bounds
                if p['type'] == 'float' or (p['type'] == 'int' and p['step'] > 1):
                    domain = [float(i) for i in p['domain']]; domain[1] -= p['add_term']
                else: domain = p['domain']
                domains.append(domain)
                self._param_combinations_generator = _gridsearch_param_combinations([1 for i in self.params], steps, domains)

            flat_params = next(self._param_combinations_generator)

        # set gridsearch params
        for i,pname in enumerate(self.params.keys()):
            p = self.params[pname]
            if p['type'] == 'float':
                p['value'] = flat_params[i] + p['add_term']
            elif p['type'] == 'int':
                p['value'] = int(flat_params[i] + p['add_term'])
            elif p['type'] == 'categorical':
                p['value'] = int(flat_params[i])

        # construct params
        new_params = {}
        for param_name in search_space:
            p = self.params[param_name]
            if p['type'] == 'float': new_params[param_name] = float(p['value'])
            elif p['type'] == 'int': new_params[param_name] = int(p['value'])
            elif p['type'] == 'categorical': new_params[param_name] = p['choices'][int(p['value'])]
        return new_params


    # The rest are unrelated to SA algorithm: boilerplate
    def infer_relative_search_space(self, study, trial):
        return optuna.search_space.intersection_search_space(study.get_trials(deepcopy=False))

    def sample_independent(self, study, trial, param_name, param_distribution):
        independent_sampler = optuna.samplers.RandomSampler()
        return independent_sampler.sample_independent(study, trial, param_name, param_distribution)