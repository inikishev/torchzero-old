"""Basic blocks"""
from typing import Optional, Any
from collections.abc import Sequence, Callable
import torch

from ..quick import seq
from .._library.norm import create_norm
from .._library.dropout import create_dropout
from .._library.pool import create_pool
from .._library.activation import create_act
from .._library.upsample import create_upsample
from .._library.flatten import create_flatten
from ..layers.pad import pad_like
from ..layers.func import func_to_named_module

_MODULE_CREATORS = {
    'A': create_act,
    'D': create_dropout,
    'N': create_norm,
    'P': create_pool,
    'U': create_upsample,
    'F': create_flatten,
}

def _create_module_order(
    modules: dict[str, Any],
    order: str,
    num_channels: Optional[int],
    ndim: Optional[int],
    spatial_size: Optional[Sequence[int]],
):
    module_order = []
    # capitalize letters
    modules = {k.upper():v for k,v in modules.items()}
    # for each letter
    for c in order.upper():
        # create module if in known module creators
        if c in _MODULE_CREATORS: mod = _MODULE_CREATORS[c](modules[c], num_channels=num_channels, ndim=ndim, spatial_size=spatial_size)
        # otherwise take from modules
        elif c in modules: mod = modules[c]
        else: raise ValueError(f"Unknown order type `{c}`")

        # set created module
        if mod is not None:
            # append module
            if isinstance(mod, torch.nn.Module): module_order.append(mod)
            # or turn into a module and append
            elif callable(mod): module_order.append(func_to_named_module(mod))
            # else what is it???
            else: raise ValueError(f"Unknown `{c}` module type `{mod}`")

    return torch.nn.Sequential(*module_order)

class GenericBlock(torch.nn.Module):
    def __init__(self,
        module: torch.nn.Module | Callable | Sequence[torch.nn.Module | Callable],
        out_channels: Optional[int] = None,
        norm: Optional[torch.nn.Module | str | bool | Callable] = None,
        dropout: Optional[float | torch.nn.Module | Callable] = None,
        act: Optional[torch.nn.Module | Callable] = None,
        pool: Optional[int | torch.nn.Module | Callable] = None,
        residual = False,
        recurrent = 1,
        ndim = 2,
        order = "MPAND",
        spatial_size: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        if callable(module): self.module = module
        else: self.module = seq(module)

        self.layers = _create_module_order(
            modules = dict(M=self.module, P=pool, A=act, N=norm, D=dropout),
            order = order,
            num_channels = out_channels,
            ndim = ndim,
            spatial_size = spatial_size,
            )

        self.residual = residual
        self.recurrent = recurrent

    def forward(self, x:torch.Tensor):
        for _ in range(self.recurrent):
            if self.residual: x = x + pad_like(self.layers(x), x)
            else: x = self.layers(x)
        return x
