from collections.abc import Sequence, Callable
from typing import Any, Literal, Optional
from functools import partial

import torch
from ....python_tools import auto_compose, identity, Compose
from ....random.random import uniform

SCALE_LITERALS = Literal[None, 'linear', 'log', 'log2', 'log10', 'ln']

SCALERS = {
    None: identity,
    'linear': identity,
    'log': torch.log10,
    'log2': torch.log2,
    'log10': torch.log10,
    'ln': torch.log
}

class XPow:
    def __init__(self, base):
        self.base = base
    def __call__(self, x):
        return self.base ** x

UNSCALERS= {
    None: identity,
    'linear': identity,
    'log': XPow(10),
    'log2': XPow(2),
    'log10': XPow(10),
    'ln': torch.exp
}

class ParamNumeric(torch.nn.Module):
    def __init__(
        self,
        name: str,
        min: Optional[float],
        max: Optional[float],
        options: Optional[dict[str, Any]],
        scale: SCALE_LITERALS,
        mul: float,
        init: Callable | float | torch.Tensor | Literal['uniform'],
        tfm: Optional[Callable | Sequence[Callable]],
        clip_params: bool,
        clip_value: bool,
    ):
        """Initialize a value.

        :param name: Value name.
        :param min: unscaled min value, only enforced if clip=True. For unconstrained you can set it to `None`.
        :param max: unscaled max value, only enforced if clip=True. For unconstrained you can set it to `None`.
        :param options: Dictionary of additional options that a `trial` will pass to the optimizer (e.g. custom lr, etc).
        :param scale: Scaling, linear or logarithmic. Optimizer will see a unscaled value (i.e. before applying log10).
        :param mul: Optimizer sees this value multiplied by `mul`. Useful to make very big values act as smaller values.
        :param init: Initialization for the value. If Callable, should be something like `torch.randn` and will get passed (1, dtype=torch.float32). If float or scalar torch.Tensor, will be initialized with that value. If "uniform", will be picked uniformly between min and max.
        :param tfm: A callable that will be applied to the scaled value (e.g. after applying log10 and dividing by `mul`).
        :param clip_params: Whether to clip params to min and max before each step. May have unintended consequences on some optimizers.
        :param clip_value: Whether to clip the value to min and max after each step.
        """

        if init == 'uniform' and (min is None or max is None): raise ValueError('min and max must be provided if init is "uniform"')
        super().__init__()
        self.name = name
        self.scaled_min, self.scaled_max = min, max
        self.scale = scale
        self.mul = mul
        self.options = options if options is not None else {}
        self.init = init

        self.scaler = SCALERS[scale]
        """Callable that applies to `self.value` and returns actual scaled value, e.g. `torch.log10`"""
        self.unscaler = UNSCALERS[scale]
        """Callable that applies to an actual scaled value and makes it unscaled."""

        self.unscaled_min = (self.unscaler(self.scaled_min) * self.mul) if self.scaled_min is not None else None
        """Actual min of `self.value`. E.g. if unscaled value is 3 and scale is log10, this will be 1000 since log10(1000) = 3"""
        self.unscaled_max = self.unscaler(self.scaled_max) * self.mul if self.scaled_max is not None else None
        """Scaled max of `self.value`. E.g. if unscaled value is 3 and scale is log10, this will be 1000 since log10(1000) = 3"""

        self.tfm = auto_compose(tfm)
        """Transforms are applied to unscaled `self.value` (keep in mind that it is a scalar tensor)"""
        self.clip_params = clip_params
        self.clip_value = clip_value

        init_min = self.unscaled_min if self.unscaled_min is not None else -1
        """min for random initialization"""
        init_max = self.unscaled_max if self.unscaled_max is not None else 1
        """max for random initialization"""
        if init_max < init_min: init_max = init_min + 1

        # initialize self.value
        # init is callable
        if callable(init):
            self.value = torch.nn.Parameter(init(1, dtype=torch.float32), requires_grad = True)
            """Scaled value of the parameter."""
        # init is a string
        elif isinstance(init, str):
            if init == 'uniform':
                self.value = torch.nn.Parameter(uniform(1, init_min, init_max, dtype=torch.float32), requires_grad = True)
            else: raise ValueError(f'Invalid init: {init}')
        # init is a value
        else:
            self.value = torch.nn.Parameter(torch.tensor(self.unscaler(init) * self.mul, dtype=torch.float32), requires_grad = True)

    def _clip_params(self):
        if self.clip_params:
            if self.unscaled_min is not None and self.value < self.unscaled_min: self.value.clamp_min_(self.unscaled_min)
            if self.unscaled_max is not None and self.value > self.unscaled_max: self.value.clamp_min_(self.unscaled_max)

    def _clip_value(self, value):
        if self.clip_params or self.clip_value:
            if self.unscaled_min is not None and value < self.unscaled_min: value = self.unscaled_min
            if self.unscaled_max is not None and value > self.unscaled_max: value = self.unscaled_max
        return value

    @torch.no_grad
    def forward(self):
        self._clip_params()
        return self._clip_value(self.tfm(self.scaler(self.value / self.mul)))

def _to_int(x): return int(round(float(x)))
class ParamInt(ParamNumeric):
    def __init__(
        self,
        name: str,
        min: Optional[float],
        max: Optional[float],
        options: Optional[dict[str, Any]],
        scale: SCALE_LITERALS,
        mul: float,
        init: Callable | float | torch.Tensor | Any,
        tfm: Optional[Callable],
        clip_params: bool,
        clip_value: bool,
    ):
        tfm = Compose(auto_compose(tfm), _to_int)
        super().__init__(
            name=name,
            min=min,
            max=max,
            options=options,
            scale=scale,
            mul=mul,
            init=init,
            tfm=tfm,
            clip_params = clip_params,
            clip_value = clip_value,
        )


class ParamFloat(ParamNumeric):
    def __init__(
        self,
        name: str,
        min: Optional[float],
        max: Optional[float],
        options: Optional[dict[str, Any]],
        scale: SCALE_LITERALS,
        mul: float,
        init: Callable | float | torch.Tensor | Literal['uniform'],
        tfm: Optional[Callable],
        clip_params: bool,
        clip_value: bool,
    ):
        tfm = Compose(auto_compose(tfm), float)
        super().__init__(
            name=name,
            min=min,
            max=max,
            options=options,
            scale=scale,
            mul=mul,
            init=init,
            tfm=tfm,
            clip_params = clip_params,
            clip_value = clip_value,
        )


def _to_bool(x): return x > 0
class ParamBool(ParamNumeric):
    def __init__(
        self,
        name: str,
        options: Optional[dict[str, Any]],
        mul: float,
        init: Callable | float | torch.Tensor | Literal['uniform'] ,
        tfm: Optional[Callable],
        clip_params: bool,
    ):
        tfm = Compose(auto_compose(tfm), _to_bool)
        super().__init__(
            name=name,
            min=-1,
            max=1,
            options=options,
            mul=mul,
            scale=None,
            init=init,
            tfm=tfm,
            clip_params = clip_params,
            clip_value = False,
        )


def _choose(x, choices:Sequence): return choices[int(x % len(choices))]
class ParamCategorical(ParamNumeric):
    def __init__(
        self,
        name: str,
        choices: Sequence,
        options: Optional[dict[str, Any]],
        mul: float,
        init: Callable | float | torch.Tensor | Literal['uniform'] | Any,
        tfm: Optional[Callable],
        clip_params: bool,
    ):
        tfm = Compose(auto_compose(tfm), partial(_choose, choices=choices))

        #if init in choices: init = choices.index(init)
        super().__init__(
            name=name,
            min=0,
            max=len(choices),
            options=options,
            mul=mul,
            scale=None,
            init=init,
            tfm=tfm,
            clip_params = clip_params,
            clip_value = False,
        )
