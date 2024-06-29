"""Basic blocks"""
from typing import Optional
from collections.abc import Sequence, Callable
import torch
from .generic_block import _create_module_order
from ..layers.pad import pad_like
from .._library.activation import create_act
from .._library.norm import create_norm
from .._library.dropout import create_dropout
__all__ = [
    'LinearBlock',
]

class LinearBlock(torch.nn.Module):
    def __init__(self,
        in_features: Optional[int],
        out_features: int,
        bias: bool = True,
        norm: Optional[torch.nn.Module | str | bool | Callable] = None,
        dropout: Optional[float | torch.nn.Module | Callable] = None,
        act: Optional[torch.nn.Module | Callable] = None,
        flatten: bool | Sequence[int] = False,
        residual = False,
        recurrent = 1,
        custom_op = None,
        order = 'FLAND'
        ):
        """Linear block

        Args:
            in_features (Optional[int]): _description_
            out_features (int): _description_
            bias (bool, optional): _description_. Defaults to True.
            norm (bool, optional): _description_. Defaults to False.
            dropout (Optional[float], optional): _description_. Defaults to None.
            act (Optional[torch.nn.Module], optional): _description_. Defaults to None.
            flatten (bool | Sequence[int], optional): _description_. Defaults to False.
            lazy (bool, optional): _description_. Defaults to False.
            custom_op (_type_, optional): _description_. Defaults to None.
            order (str, optional): _description_. Defaults to 'fland'.
        """
        super().__init__()
        # linear
        if in_features is None: self.linear = torch.nn.LazyLinear(out_features, bias)
        elif custom_op is None: self.linear = torch.nn.Linear(in_features, out_features, bias) # type:ignore
        else: self.linear = custom_op(in_features, out_features, bias) # type:ignore

        self.layers = _create_module_order(
            modules = dict(L=self.linear, A=act, N=norm, D=dropout, F=flatten),
            order = order,
            main_module='L',
            in_channels = in_features,
            out_channels= out_features,
            ndim = 0,
            spatial_size = None,
            )

        self.residual = residual
        self.recurrent = recurrent

    def forward(self, x:torch.Tensor):
        for _ in range(self.recurrent):
            if self.residual: x = x + pad_like(self.layers(x), x)
            else: x = self.layers(x)
        return x
