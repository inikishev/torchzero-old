"""Basic optimizer combination, which tends to work poorly on gradient-based optimizers."""
from functools import partial
import random
from typing import Optional
from collections.abc import Callable, Sequence
import torch
from torch.optim import Optimizer

from .utils import foreach_param, foreach_group_param

class SequentialOpt(Optimizer):
    def __init__(self, params, optimizers:Sequence):
        """Sequentially apply all optimizers each step. Probably best not to use with momentum-based optimizers, and most gradient-based optimizers."""
        super().__init__(params, {})
        self.optimizers = optimizers

    @torch.no_grad
    def step(self, closure: Optional[Callable] = None):# type:ignore #pylint:disable=W0222
        loss = None
        for optimizer in self.optimizers:
            optimizer.zero_grad()
            optimizer_loss = optimizer.step(closure)
            if optimizer_loss is not None: loss = optimizer_loss

        return loss

class RandomOpt(Optimizer):
    def __init__(self, params, optimizers:Sequence, weights:Optional[Sequence[float]] = None):
        """Selects a random optimizer each step. Weights is if you want random choice to be weighted. Probably best not to use with momentum-based optimizers, and most gradient-based optimizers."""
        super().__init__(params, {})
        self.optimizers = optimizers
        if weights is None: weights = [1] * len(optimizers)
        self.weights = weights

    @torch.no_grad
    def step(self, closure: Optional[Callable] = None):# type:ignore #pylint:disable=W0222
        optimizer = random.choices(self.optimizers, weights=self.weights)[0]
        return optimizer.step(closure)

class AlternatingOpt(Optimizer):
    def __init__(self, params, optimizers:Sequence):
        """Alternates between all optimizers, once optimizer per step. Probably best not to use with momentum-based optimizers, and most gradient-based optimizers."""
        super().__init__(params, {})
        self.optimizers = optimizers
        self.cur_step = 0

    @torch.no_grad
    def step(self, closure: Optional[Callable] = None):# type:ignore #pylint:disable=W0222
        optimizer = self.optimizers[self.cur_step % len(self.optimizers)]
        self.cur_step += 1
        return optimizer.step(closure)

class PreserveGrad(Optimizer):
    def __init__(self, params, optimizers:Sequence):
        """Preserves gradients for gradient approximation based optimizers."""
        super().__init__(params, {})
        self.optimizers = optimizers
        self.cur_step = 0

    @torch.no_grad
    def step(self, closure: Optional[Callable] = None):# type:ignore #pylint:disable=W0222
        optimizer = self.optimizers[self.cur_step % len(self.optimizers)]
        self.cur_step += 1
        return optimizer.step(closure)


class OptimizerAverage(Optimizer):
    def __init__(self, params, optimizers:Sequence, weights:Optional[list[float]] = None, loss_weighted = True, noise=None, sampler = torch.randn_like, nbest=None):
        """Makes a step with each optimizer on a copy of parameters and then takes the average update. `nbest` takes average of n best optimizers. If you set `nbest` to 1, it will take the best update. Probably best not to use with momentum-based optimizers, and most gradient-based optimizers."""
        defaults = dict(noise = noise, sampler = sampler)
        super().__init__(params, defaults)
        self.optimizers = optimizers
        if weights is None: weights = [1 for _ in range(len(optimizers))]
        self.weights = weights
        self.loss_weighted = loss_weighted
        self.nbest = nbest

    @torch.no_grad
    def step(self, closure: Optional[Callable] = None):# type:ignore #pylint:disable=W0222
        # save initial params
        for group, p in foreach_group_param(self.param_groups):
            state = self.state[p]
            state['before'] = p.clone()

        losses = {}

        # make a step with each optimizer from initial params
        for i, opt in enumerate(self.optimizers):
            # copy initial params into parameters
            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]
                p.copy_(state['before'])
                # add noise to parameters
                if (group["noise"] is not None) and (group["noise"] != 0):
                    p.add_(group["sampler"](p) * group["noise"])

            # make a step with initial parameters
            losses[i] = opt.step(closure)

            # calculate the update for current optimizer
            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]
                state[f"{i} update"] = p - state["before"]

        # get normalized weights
        # get best n losses and weighted
        if self.nbest is None: nbest = len(losses)
        else: nbest = self.nbest
        weighted_losses = {k:v * self.weights[k] for k,v in sorted(losses.items(), key=lambda item: item[1])[:nbest]}
        # normalize
        weighted_loss_min = min(list(weighted_losses.values()))
        weighted_losses = {k: v - weighted_loss_min for k,v in weighted_losses.items()}
        weighted_loss_sum = sum(list(weighted_losses.values()))
        if weighted_loss_sum == 0: normalized_weighted_losses = {k: 1/len(weighted_losses) for k in weighted_losses}
        else: normalized_weighted_losses = {k: v / weighted_loss_sum for k,v in weighted_losses.items()}



        # make a step with averaged update
        for group, p in foreach_group_param(self.param_groups):
            state = self.state[p]
            # put initial state into parameter and add each optimizers update multiplied by its normalized weight
            p.copy_(state['before'])
            for i in range(len(self.optimizers)):
                if i in normalized_weighted_losses:
                    p.add_(state[f'{i} update'] * normalized_weighted_losses[i])

        return min(list(losses.values()))
