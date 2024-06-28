from collections.abc import Sequence
import torch
from .layers.func import ensure_module
def seq(layers, *args):
    """Shortcut for torch.nn.Sequential that also converts non module callables into modules."""
    modules = []
    if isinstance(layers, Sequence): modules.extend([ensure_module(i) for i in layers])
    else: modules.append(ensure_module(layers))
    modules.extend([ensure_module(i) for i in args])
    return torch.nn.Sequential(*modules)
