"""Basic blocks"""
from typing import Optional
from collections.abc import Sequence, Callable
import torch
from .generic_block import _create_module_order
from .pad import pad_like
from .sequential import Sequential

from .._library.activation import create_act
from .._library.norm import create_norm
from .._library.dropout import create_dropout
__all__ = [
    'LinearBlock',
]

class LinearBlock(Sequential):
    def __init__(self,
        in_features: Optional[int],
        out_features: int,
        bias: bool = True,
        norm: Optional[torch.nn.Module | str | bool | Callable] = None,
        dropout: Optional[float | torch.nn.Module | Callable] = None,
        act: Optional[torch.nn.Module | Callable] = None,
        flatten: bool | Sequence[int] = False,
        custom_op = None,
        order = 'FLAND',
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
        # linear
        if in_features is None: self.linear = torch.nn.LazyLinear(out_features, bias)
        elif custom_op is None: self.linear = torch.nn.Linear(in_features, out_features, bias) # type:ignore
        else: self.linear = custom_op(in_features, out_features, bias) # type:ignore

        layers = _create_module_order(
            modules = dict(L=self.linear, A=act, N=norm, D=dropout, F=flatten),
            order = order,
            main_module='L',
            in_channels = in_features,
            out_channels= out_features,
            ndim = 0,
            spatial_size = None,
            )


        super().__init__(*layers)
