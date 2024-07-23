from functools import partial
from typing import Optional
from collections.abc import Callable, Sequence
import torch
from torch.optim import Optimizer
import numpy as np

from .utils import foreach_param, foreach_group_param
from .genetic.crossover import crossover_swap, crossover_uniform_, crossover_onepoint_, crossover_mean, crossover_random_strat
from ..random import similar_like

DEFAULT_CROSSOVER_STRATS = [crossover_swap, crossover_uniform_, crossover_onepoint_, crossover_mean]
DEFAULT_CROSSOVER_WEIGHTS = [8, 1, 4, 4]
DEFAULT_CROSSOVER = partial(crossover_random_strat, strats=DEFAULT_CROSSOVER_STRATS, weights=DEFAULT_CROSSOVER_WEIGHTS)


class SwarmOfOptimizers(Optimizer):
    def __init__(self, 
                 initializer: Callable[[], tuple], 
                 num:int = 8, # should be schedulable
                 crossover_p = 0.75,
                 crossover_strat = DEFAULT_CROSSOVER,
                 death_age = 100,
                 steps_per_batch = 4,
                 
                 ): ...