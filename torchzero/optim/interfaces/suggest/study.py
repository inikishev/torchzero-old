from collections.abc import Callable
import torch
from .trial import Trial

class Study:
    def __init__(self, objective: Callable[[Trial], float | torch.Tensor]):
        """Initialize a study.

        :param objective: A callable that accepts `Trial` object as the first argument.

        Examples:

        ```py
        # define an objective
        def objective(trial: Trial):
            x = trial.suggest_float('x', -10, 10)
            y = trial.suggest_float('y', -10, 10)
            return (x - 3) ** y

        # create a study
        study = Study(objective)
        # create an optimizer
        optimizer = AcceleratedRandomSearchSampler(study)

        # optimize for 100 steps
        for step in range(100):
            closure = study.step
            loss = optimizer.step(closure)

        # get best params
        best_params = study.best_params

        ```
        """
        self.objective = objective
        self.trial = Trial()
        self.best_value = self.objective(self.trial)

        self.loss_history = []
        self.param_history = []
        self.best_value = float('inf')
        self.best_params = None

    def parameters(self):
        """Return PyTorch parameters of the trial that can be passed to an optimizer, includes any custom optimizer options that were passed to suggested values."""
        params_without_options = {"params": []}
        params_with_options = []
        for param in self.trial.children():
            if len(param.options) == 0: params_without_options["params"].extend(param.parameters())
            else: params_with_options.append({'params': param.parameters(), **param.options})

        return params_with_options + [params_without_options]

    def get_values(self):
        """Return current trial values as {name: value}"""
        return {mod.name: mod() for mod in self.trial.children()}

    def step(self):
        """Run objective with current parameters and return the objective value."""
        loss = self.objective(self.trial)

        if isinstance(loss, torch.Tensor): float_loss = float(loss.detach().cpu())
        else: float_loss = float(loss)
        self.loss_history.append(float_loss)

        values = self.get_values()
        self.param_history.append(values)
        if float_loss < self.best_value:
            self.best_value = float_loss
            self.best_params = values

        return loss

    def __call__(self): return self.step()

