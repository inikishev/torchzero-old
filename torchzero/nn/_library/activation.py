from collections.abc import Sequence
from typing import Optional, Any
import torch
from ..layers.act.sine import SineAct
from ..layers.act.piecewise import VShaped
from ..layers.act.relu import *
from ..layers.act.selft import *

def identity(x): return x

def create_act(module:Any, num_channels:Optional[int], ndim:Optional[int], spatial_size:Optional[Sequence[int]]):
    if module is None: return None
    if callable(module): return module

    if isinstance(module, str): module = ''.join([i for i in module.lower().strip() if i.isalnum()])

    if module in ('relu',): return torch.nn.ReLU(inplace=True)
    if module in ('leakyrelu',): return torch.nn.LeakyReLU(0.1, inplace=True)
    if module in ('prelu',): return torch.nn.PReLU(num_parameters=num_channels if num_channels is not None else 1)
    if module in ('rrelu',): return torch.nn.RReLU(inplace=True)
    if module in ('silu',): return torch.nn.SiLU(inplace=True)
    if module in ('celu',): return torch.nn.CELU(inplace=True)
    if module in ('gelu',): return torch.nn.GELU()
    if module in ('selu',): return torch.nn.SELU(inplace=True)
    if module in ('elu',): return torch.nn.ELU(inplace=True)
    if module in ('relu6'): return torch.nn.ReLU6(inplace=True)
    if module in ('mish'): return torch.nn.Mish(inplace=True)
    if module in ('hardswish',): return torch.nn.Hardswish(inplace=True)
    if module in ('softplus',): return torch.nn.Softplus()
    if module in ('threshold',): return torch.nn.Threshold(0, -1, inplace=True)

    if module in ('sigmoid',): return torch.nn.Sigmoid()
    if module in ('tanh',): return torch.nn.Tanh()
    if module in ('softmax',): return torch.nn.Softmax(dim=1)
    if module in ('hardtanh',): return torch.nn.Hardtanh()
    if module in ('logsoftmax',): return torch.nn.LogSoftmax(dim=1)
    if module in ('softshrink',): return torch.nn.Softshrink()
    if module in ('hardshrink',): return torch.nn.Hardshrink()
    if module in ('tanhshrink',): return torch.nn.Tanhshrink()
    if module in ('hardsigmoid',): return torch.nn.Hardsigmoid()
    if module in ('logsigmoid',): return torch.nn.LogSigmoid()
    if module in ('softmin',): return torch.nn.Softmin(dim=1)
    if module in ('softsign',): return torch.nn.Softsign(dim=1)

    if module in ('genrelu', 'generalrelu'): return GeneralReLU(leak = 0.1, sub = 0.3, maxv = 10, inplace=True)
    if module in ('symrelu', 'symmetricrelu'): return SymReLU(inplace=True, maxv=10)
    if module in ('lerelu','learnablerelu'): return LearnReLU()
    if module in ('legenrelu', 'learnablegeneralrelu'): return LearnReLU()
    if module in ('leleakyrelu', 'learnableleakyrelu'): return LearnLeakyReLU()
    if module in ('lesymrelu', 'learnablesymmetricrelu'): return LearnSymReLU()
    if module in ('halfminmax',): return HalfMinMax()
    if module in ('halfsummax'): return HalfSumMax()
    if module in ('halfmulmax'): return HalfMulMax()
    if module in ('firstmax'): return FirstMax()

    if module in ('sin','sine'): return SineAct()
    if module in ('v','vshaped',): return VShaped()

    if module in ('identity',): return identity
    else: raise ValueError(f'Unknown activation module: {module}')
