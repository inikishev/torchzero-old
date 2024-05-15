from typing import Optional
from collections.abc import Callable
import torch
from torch.optim import Optimizer

from .utils import foreach_param, foreach_group_param
