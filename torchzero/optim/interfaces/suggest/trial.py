from collections.abc import Callable, Sequence
from typing import Optional, Any, Literal

import torch

from .parameters import ParamBool, ParamCategorical, ParamFloat, ParamInt
from .parameters import SCALE_LITERALS

class Trial(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def suggest_float(
        self,
        name: str,
        min: Optional[float] = None,
        max: Optional[float] = None,
        options: Optional[dict[str, Any]] = None,
        scale: SCALE_LITERALS = None,
        mul=1.0,
        init: Callable | float | torch.Tensor | Literal['uniform'] = 'uniform',
        tfm=None,
        clip_params:bool = True,
        clip_value:bool = False,
    ):
        """Suggest a value for the floating point parameter. If this value has already been suggested, returns the value, so all of the options except `name` will be ignored.

        :param name: Name of the value.
        :param min: Lowest endpoint of the range of suggested values. Some samplers will ignore this, use `clip_param` or `clip_value` to enforce it. If None, no minimum will be enforced.
        :param max: Highest endpoint of the range of suggested values. Some samplers will ignore this, use `clip_param` or `clip_value` to enforce it. If None, no maximum will be enforced.
        :param options: Dictionary of per-parameter optimizer options for this parameter. defaults to None
        :param scale: Scale. Optimizer will see a unscaled value (e.g. before applying log10). defaults to None
        :param mul: To optimizer this value will appear multiplied by `mul`. Useful to make very big values act as smaller values.
        :param init: Initialization for the value. If Callable, should be something like `torch.randn` and will get passed (1, dtype=torch.float32). If float or scalar torch.Tensor, will be initialized with that value. If "uniform", will be picked uniformly between min and max (if those are None, will be picked between -1 and 1).
        :param tfm: A callable that will be applied to the scaled value (e.g. after applying log10 and dividing by `mul`).
        :param clip_params: Whether to clip params to min and max before each step. May have unintended consequences on some optimizers.
        :param clip_value: Whether to clip the returned value to min and max after each step.
        :return: suggested value for the parameter.
        """
        if name not in dict(self.named_modules()):
            self.add_module(name, ParamFloat(
                name = name,
                min = min,
                max = max,
                options = options,
                scale = scale,
                mul = mul,
                init = init,
                tfm=tfm,
                clip_params = clip_params,
                clip_value = clip_value,
                )
            )
        return getattr(self, name)()

    def suggest_int(
        self,
        name: str,
        min: Optional[float] = None,
        max: Optional[float] = None,
        options: Optional[dict[str, Any]] = None,
        scale: SCALE_LITERALS = None,
        mul=0.01,
        init: Callable | float | torch.Tensor | Literal['uniform']  = 'uniform',
        tfm=None,
        clip_params:bool = True,
        clip_value:bool = True,
    ):
        """Suggest a value for the integer parameter. Note that unlike in Optuna, to the optimizer this is still a float value, and just gets rounded to an integer. If this value has already been suggested, returns the value, so all of the options except `name` will be ignored.

        :param name: Name of the value.
        :param min: Lowest endpoint of the range of suggested values. Some samplers will ignore this, use `clip_param` or `clip_value` to enforce it.
        :param max: Highest endpoint of the range of suggested values. Some samplers will ignore this, use `clip_param` or `clip_value` to enforce it.
        :param options: Dictionary of per-parameter optimizer options for this parameter. defaults to None
        :param scale: Scale. Optimizer will see a unscaled value (e.g. before applying log10). defaults to None
        :param mul: To optimizer this value will appear multiplied by `mul`. Useful to make smaller changes to the value actually reflect after converting it to an integer. For example, if `mul` is 0.01, then optimizer will see 0.15, which will be converted to 15.
        :param init: Initialization for the value. If Callable, should be something like `torch.randn` and will get passed (1, dtype=torch.float32). If float or scalar torch.Tensor, will be initialized with that value. If "uniform", will be picked uniformly between min and max.
        :param tfm: A callable that will be applied to the scaled value (e.g. after applying log10 and dividing by `mul`), but before converting it to an integer.
        :param clip_params: Whether to clip params to min and max before each step. May have unintended consequences on some optimizers.
        :param clip_value: Whether to clip the returned value to min and max after each step.
        :return: suggested value for the parameter.

        :return: _description_
        """
        if name not in dict(self.named_modules()):
            self.add_module(name, ParamInt(
                name = name,
                min = min,
                max = max,
                options = options,
                scale = scale,
                mul = mul,
                init = init,
                tfm=tfm,
                clip_params = clip_params,
                clip_value = clip_value,
                )
            )
        return getattr(self, name)()

    def suggest_bool(
        self,
        name: str,
        options: Optional[dict[str, Any]] = None,
        mul=0.01,
        init: Callable | float | torch.Tensor | Literal['uniform'] = 0.,
        tfm=None,
        clip_params = True,
    ):
        """Suggest a value for the boolean parameter (True or False). Note that to the optimizer this is still a float value, and it gets converted to True if it is higher than 0. If this value has already been suggested, returns the value, so all of the options except `name` will be ignored.

        :param name: Name of the value.
        :param options: Dictionary of per-parameter optimizer options for this parameter. defaults to None
        :param mul: To optimizer this value will appear multiplied by `mul`. Useful to make smaller changes to the value actually reflect after converting it to a boolean.
        :param init: Initial float value. Defaults to 0.
        :param tfm: A callable that will be applied to the scaled value (e.g. after applying log10 and dividing by `mul`), but before converting it to a boolean.
        :param clip_params: Whether to clip params to min and max before each step. May have unintended consequences on some optimizers.
        :return: suggested value for the parameter.
        """
        if name not in dict(self.named_modules()):
            self.add_module(name, ParamBool(
                name = name,
                options = options,
                mul = mul,
                init = init,
                tfm=tfm,
                clip_params = clip_params,
                )
            )
        return getattr(self, name)()

    def suggest_categorical(
        self,
        name: str,
        choices: Sequence[Any],
        options: Optional[dict[str, Any]] = None,
        mul=0.01,
        init: Callable | float | torch.Tensor | Literal['uniform'] | Any = 'uniform',
        tfm=None,
        clip_params: bool = False,
    ):
        """Suggest a value for the boolean parameter (True or False). Note that to the optimizer this is still a float value, and it gets converted to a rolling integer index and used to index `choices`. If this value has already been suggested, returns the value, so all of the options except `name` will be ignored.

        :param name: Name of the value.
        :param choices: A sequence of all values that can be chosen.
        :param options: Dictionary of per-parameter optimizer options for this parameter. defaults to None
        :param mul: To optimizer this value will appear multiplied by `mul`. Useful to make smaller changes to the value actually reflect after converting it to an integer index. For example, if `mul` is 0.01, then optimizer will see 0.03, which will be converted to 3 and 3rd item will be selected.
        :param init: Initial index. If Callable, should be something like `torch.randn` and will get passed (1, dtype=torch.float32). If float or scalar torch.Tensor, will be initialized with that value. If "uniform", will be picked uniformly between min and max (if those are None, will be picked between -1 and 1).
        :param tfm: A callable that will be applied to the scaled value (e.g. after applying log10 and dividing by `mul`), but before converting it to a boolean.
        :param clip_params: Whether to clip params to min and max before each step. May have unintended consequences on some optimizers.
        :return: suggested value for the parameter.
        """
        if name not in dict(self.named_modules()):
            self.add_module(name, ParamCategorical(
                name = name,
                choices = choices,
                options = options,
                mul = mul,
                init = init,
                tfm=tfm,
                clip_params = clip_params,
                )
            )
        return getattr(self, name)()

    def forward(self): raise NotImplementedError('Trial class is not callable. Use suggest_* methods to define parameters.')

