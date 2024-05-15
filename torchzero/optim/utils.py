from collections.abc import Iterable
import torch
from torch.optim import Optimizer

def foreach_param(param_groups: list[dict[str, list[torch.nn.Parameter]]]) -> Iterable[torch.nn.Parameter]:
    """Iterates over all learnable parameters in param_groups"""
    for group in param_groups:
        for param in group["params"]:
            if param.requires_grad is True: yield param

def foreach_group_param(param_groups: list[dict[str, list[torch.nn.Parameter]]]) -> Iterable[tuple[dict, torch.nn.Parameter]]:
    """Iterates over all learnable parameters in param_groups, returning their group"""
    for group in param_groups:
        for param in group["params"]:
            if param.requires_grad is True: yield group, param

def foreach_param_in_group(group: dict[str, list[torch.nn.Parameter]]) -> Iterable[torch.nn.Parameter]:
    for param in group["params"]:
        if param.requires_grad is True: yield param