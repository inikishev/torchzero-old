# pyright: reportCallIssue = false, reportArgumentType = false
#region imports
import random
from collections.abc import Callable, Sequence, Iterable
from typing import Protocol, Optional
from typing import Any
import torch
from torch import Tensor
from typing_extensions import TypeVar
from ..random.random import rademacher as _rademacher, rademacher_like as _rademacher_like
#endregion

#region types
TensorSequence = list[Tensor] | tuple[Tensor, ...] | list[torch.nn.Parameter] | tuple[torch.nn.Parameter, ...]
PyNumber = int | float | bool
ScalarOrTensorOrAnySequence = Tensor | torch.nn.Parameter | TensorSequence | PyNumber | Sequence[PyNumber] | Any
ScalarOrAnySequence = PyNumber | TensorSequence | Any
#endregion

#region functions
def add(self:TensorSequence, other:ScalarOrTensorOrAnySequence, *, alpha:PyNumber = 1, foreach:bool) -> TensorSequence:
    if foreach:
        if alpha == 1: return torch._foreach_add(self, other)
        return torch._foreach_add(self, other, alpha=alpha)
    return [torch.add(t1, t2, alpha=alpha) for t1, t2 in zip(self, other)]
def add_(self:TensorSequence, other:ScalarOrTensorOrAnySequence, *, alpha:PyNumber = 1, foreach:bool) -> None:
    if foreach:
        if alpha == 1: return torch._foreach_add_(self, other)
        return torch._foreach_add_(self, other, alpha=alpha)
    for t1, t2 in zip(self, other): t1.add_(t2, alpha=alpha)


def sub(self:TensorSequence, other:ScalarOrTensorOrAnySequence, *, alpha:PyNumber = 1, foreach:bool) -> TensorSequence:
    if foreach:
        if alpha == 1: return torch._foreach_sub(self, other)
        return torch._foreach_sub(self, other, alpha=alpha)
    return [torch.sub(t1, t2, alpha=alpha) for t1, t2 in zip(self, other)]
def sub_(self:TensorSequence, other:ScalarOrTensorOrAnySequence, *, alpha:PyNumber = 1, foreach:bool) -> None:
    if foreach:
        if alpha == 1: return torch._foreach_sub_(self, other)
        return torch._foreach_sub_(self, other, alpha=alpha)
    for t1, t2 in zip(self, other): t1.sub_(t2, alpha=alpha)

def mul(self:TensorSequence, other:ScalarOrTensorOrAnySequence, *, foreach:bool) -> TensorSequence:
    if foreach: return torch._foreach_mul(self, other)
    if isinstance(other, (int, float, bool, Tensor)): other = [other for _ in self]
    return [torch.mul(t1, t2) for t1, t2 in zip(self, other)]
def mul_(self:TensorSequence, other:ScalarOrTensorOrAnySequence, *, foreach:bool) -> None:
    if foreach: return torch._foreach_mul_(self, other)
    if isinstance(other, (int, float, bool, Tensor)): other = [other for _ in self]
    for t1, t2 in zip(self, other): t1.mul_(t2)

def div(self:TensorSequence, other:ScalarOrTensorOrAnySequence, *, foreach:bool) -> TensorSequence:
    if foreach: return torch._foreach_div(self, other)
    if isinstance(other, (int, float, bool, Tensor)): other = [other for _ in self]
    return [torch.div(t1, t2) for t1, t2 in zip(self, other)]
def div_(self:TensorSequence, other:ScalarOrTensorOrAnySequence, *, foreach:bool) -> None:
    if foreach: return torch._foreach_div_(self, other)
    if isinstance(other, (int, float, bool, Tensor)): other = [other for _ in self]
    for t1, t2 in zip(self, other): t1.div_(t2)

def pow(self:TensorSequence, other:ScalarOrAnySequence, *, foreach:bool) -> TensorSequence:
    if foreach: return torch._foreach_pow(self, other)
    if isinstance(other, (int, float, bool)): other = [other for _ in self]
    return [torch.pow(t1, t2) for t1, t2 in zip(self, other)]
def pow_(self:TensorSequence, other:ScalarOrAnySequence, *, foreach:bool) -> None:
    if foreach: return torch._foreach_pow_(self, other)
    if isinstance(other, (int, float, bool)): other = [other for _ in self]
    for t1, t2 in zip(self, other): t1.pow_(t2)

def exp(self:TensorSequence, *, foreach:bool) -> TensorSequence:
    if foreach: return torch._foreach_exp(self)
    return [t.exp() for t in self]
def exp_(self:TensorSequence, *, foreach:bool) -> None:
    if foreach: return torch._foreach_exp_(self)
    for t in self: t.exp_()

def sign(self:TensorSequence, *, foreach:bool) -> TensorSequence:
    if foreach: return torch._foreach_sign(self)
    return [torch.sign(t) for t in self]
def sign_(self:TensorSequence, *, foreach:bool) -> None:
    if foreach: return torch._foreach_sign_(self)
    for t in self: t.sign_()

def neg(self:TensorSequence, *, foreach:bool) -> TensorSequence:
    if foreach: return torch._foreach_neg(self)
    return [torch.neg(t) for t in self]
def neg_(self:TensorSequence, *, foreach:bool) -> None:
    if foreach: return torch._foreach_neg_(self)
    for t in self: t.neg_()

def copy_(self:TensorSequence, other:TensorSequence, *, foreach:bool) -> None:
    if foreach: return torch._foreach_copy_(self, other)
    for t1, t2 in zip(self, other): t1.copy_(t2)

def softsign(self:TensorSequence, *, foreach:bool) -> TensorSequence:
    if foreach: return torch._foreach_div(self, torch._foreach_add(torch._foreach_abs(self), 1))
    return [torch.nn.functional.softsign(t) for t in self]

def set_(self:TensorSequence, other:TensorSequence, *, foreach:bool) -> None:
    # no foreach
    for t1, t2 in zip(self, other): t1.set_(t2)

def assigndata_(self:TensorSequence, other:TensorSequence, *, foreach:bool) -> None:
    # no foreach
    for t1, t2 in zip(self, other): t1.data = t2

def lgamma(self:TensorSequence, *, foreach:bool) -> TensorSequence:
    if foreach: return torch._foreach_lgamma(self)
    return [torch.lgamma(t) for t in self]
def lgamma_(self:TensorSequence, *, foreach:bool) -> None:
    if foreach: return torch._foreach_lgamma_(self)
    for t in self: t.lgamma_()

def abs(self:TensorSequence, *, foreach:bool) -> TensorSequence:
    if foreach: return torch._foreach_abs(self)
    return [torch.abs(t) for t in self]

def abs_(self:TensorSequence, *, foreach:bool) -> None:
    if foreach: return torch._foreach_abs_(self)
    for t in self: t.abs_()

def clamp_min(self:TensorSequence, other:ScalarOrAnySequence, *, foreach:bool):
    if foreach: return torch._foreach_clamp_min(self, other)
    if isinstance(other, (int, float, bool)): other = [other for _ in self]
    return [torch.clamp(t1, min=t2) for t1, t2 in zip(self, other)]
def clamp_min_(self:TensorSequence, other:ScalarOrAnySequence, *, foreach:bool):
    if foreach: return torch._foreach_clamp_min_(self, other)
    if isinstance(other, (int, float, bool)): other = [other for _ in self]
    for t1, t2 in zip(self, other): t1.clamp_min_(t2)

def clamp_max(self:TensorSequence, other:ScalarOrAnySequence, *, foreach:bool):
    if foreach: return torch._foreach_clamp_max(self, other)
    if isinstance(other, (int, float, bool)): other = [other for _ in self]
    return [torch.clamp(t1, max=t2) for t1, t2 in zip(self, other)]
def clamp_max_(self:TensorSequence, other:ScalarOrAnySequence, *, foreach:bool):
    if foreach: return torch._foreach_clamp_max_(self, other)
    if isinstance(other, (int, float, bool)): other = [other for _ in self]
    for t1, t2 in zip(self, other): t1.clamp_max_(t2)

def rand_like(self:TensorSequence, *, foreach:bool) -> TensorSequence:
    if foreach:
        vec = torch.rand(sum([t.numel() for t in self]), device = self[0].device, dtype=self[0].dtype)
        res = []
        cur_idx = 0
        for t in self:
            numel = t.numel()
            res.append(vec[cur_idx : cur_idx + numel].reshape_as(t))
            cur_idx += numel
        return res
    else: return [torch.rand_like(t) for t in self]

def randn_like(self:TensorSequence, *, foreach:bool) -> TensorSequence:
    if foreach:
        vec = torch.randn(sum([t.numel() for t in self]), device = self[0].device, dtype=self[0].dtype)
        res = []
        cur_idx = 0
        for t in self:
            numel = t.numel()
            res.append(vec[cur_idx : cur_idx + numel].reshape_as(t))
            cur_idx += numel
        return res
    else: return [torch.randn_like(t) for t in self]

def rademacher_like(self:TensorSequence, *, foreach:bool) -> TensorSequence:
    if foreach:
        vec = _rademacher(sum([t.numel() for t in self]), device = self[0].device, dtype=self[0].dtype)
        res = []
        cur_idx = 0
        for t in self:
            numel = t.numel()
            res.append(vec[cur_idx : cur_idx + numel].reshape_as(t))
            cur_idx += numel
        return res
    else: return [_rademacher_like(t) for t in self]

def fn_like(self:TensorSequence, fn:Callable, *, foreach:bool) -> TensorSequence:
    if foreach:
        vec = fn(sum([t.numel() for t in self]), device = self[0].device, dtype=self[0].dtype)
        res = []
        cur_idx = 0
        for t in self:
            numel = t.numel()
            res.append(vec[cur_idx : cur_idx + numel].reshape_as(t))
            cur_idx += numel
        return res
    else: return [fn(t.shape, dtype=t.dtype, device=t.device) for t in self]


def zero_(self:TensorSequence, *, foreach:bool) -> None:
    if foreach: return torch._foreach_zero_(self)
    for t in self: t.zero_()

def sequencemean(sequences:Sequence["TensorList"], *, foreach:bool) -> "TensorList":
    """Only use if you don't care about `sequences`! As this reuses storage of the first element."""
    if foreach:
        total = sequences[0]
        if len(sequences) > 1:
            for i in sequences[1:]:
                total.add_(i)
            total.div_(len(sequences))
        return total
    else: return TensorList([torch.mean(torch.stack(p, dim=0), dim=0) for p in zip(*sequences)], foreach=False)

def sequencesum(sequences:Sequence["TensorList"], *, foreach:bool) -> "TensorList":
    """Only use if you don't care about `sequences`! As this reuses storage of the first element."""
    if foreach:
        total = sequences[0]
        if len(sequences) > 1:
            for i in sequences[1:]:
                total.add_(i)
        return total
    else: return TensorList([torch.sum(torch.stack(p, dim=0), dim=0) for p in zip(*sequences)], foreach=False)

#region TensorList
class TensorList(list[Tensor]):
    def __init__(self, tensors:Iterable[torch.Tensor], foreach:bool) -> None:
        super().__init__(tensors)
        self.foreach = foreach

    # no!!!!!!!!!
    def __add__(self, do_not_use): # type:ignore
        print('no no no ！ This is a list method __add__. And it is not allowed!!!')
        return super().__add__(do_not_use)

    def __iadd__(self, do_not_use):
        print('no no no ！ This is a list method __iadd__. And it is not allowed!!!')
        return super().__iadd__(do_not_use)

    def __mul__(self, do_not_use):
        print('no no no ！ This is a list method __mul__. And it is not allowed!!!')
        return super().__mul__(do_not_use)

    def __imul__(self, do_not_use):
        print('no no no ！ This is a list method __imul__. And it is not allowed!!!')
        return super().__imul__(do_not_use)

    # overloads
    def __sub__(self, other:ScalarOrAnySequence): return self.sub(other)
    def __isub__(self, other:ScalarOrAnySequence): self.sub_(other); return self
    def __rsub__(self, other:ScalarOrAnySequence): return self.neg().add(other)

    def __truediv__(self, other:ScalarOrTensorOrAnySequence): return self.div(other)
    def __itruediv__(self, other:ScalarOrTensorOrAnySequence): self.div_(other); return self
    def __rtruediv__(self, other:ScalarOrTensorOrAnySequence): return self.pow(-1.).mul(other)


    def __pow__(self, other:ScalarOrAnySequence): return self.pow(other)
    def __ipow__(self, other:ScalarOrAnySequence): self.pow_(other); return self

    def __neg__(self): return self.neg()

    # basic methods
    def set_(self, other:Iterable[Tensor]):
        for t1, t2 in zip(self, other): t1.set_(t2)

    def assigndata_(self, other:Iterable[Tensor]):
        for t1, t2 in zip(self, other): t1.data = t2

    def copy_(self, other:TensorSequence): copy_(self, other, foreach=self.foreach)

    def clone(self): return TensorList([t.clone() for t in self], foreach=self.foreach)

    def zero_(self): zero_(self, foreach=self.foreach)

    # arithmetic
    def add(self, other:ScalarOrAnySequence, alpha=1.): return TensorList(add(self, other, alpha=alpha, foreach=self.foreach), foreach=self.foreach)
    def add_(self, other:ScalarOrAnySequence, alpha=1.): add_(self, other, alpha=alpha, foreach=self.foreach)

    def sub(self, other:ScalarOrAnySequence, alpha=1.): return TensorList(sub(self, other, alpha=alpha, foreach=self.foreach), foreach=self.foreach)
    def sub_(self, other:ScalarOrAnySequence, alpha=1.): sub_(self, other, alpha=alpha, foreach=self.foreach)

    def mul(self, other:ScalarOrTensorOrAnySequence): return TensorList(mul(self, other, foreach=self.foreach), foreach=self.foreach)
    def mul_(self, other:ScalarOrTensorOrAnySequence): mul_(self, other, foreach=self.foreach)

    def div(self, other:ScalarOrTensorOrAnySequence): return TensorList(div(self, other, foreach=self.foreach), foreach=self.foreach)
    def div_(self, other:ScalarOrTensorOrAnySequence): div_(self, other, foreach=self.foreach)

    def pow(self, other:ScalarOrAnySequence): return TensorList(pow(self, other, foreach=self.foreach), foreach=self.foreach)
    def pow_(self, other:ScalarOrAnySequence): pow_(self, other, foreach=self.foreach)

    def exp(self): return TensorList(exp(self, foreach=self.foreach), foreach=self.foreach)
    def exp_(self): exp_(self, foreach=self.foreach)

    def neg(self): return TensorList(neg(self, foreach=self.foreach), foreach=self.foreach)
    def neg_(self): neg_(self, foreach=self.foreach)

    def sign(self): return TensorList(sign(self, foreach=self.foreach), foreach=self.foreach)
    def sign_(self): sign_(self, foreach=self.foreach)

    def lgamma(self): return TensorList(lgamma(self, foreach=self.foreach), foreach=self.foreach)
    def lgamma_(self): lgamma_(self, foreach=self.foreach)

    def abs(self): return TensorList(abs(self, foreach=self.foreach), foreach=self.foreach)
    def abs_(self): abs_(self, foreach=self.foreach)

    def sum(self): return sum([i.sum() for i in self])
    def mean(self): return sum([i.mean() for i in self]) / len(self)
    def max(self): return max([i.max() for i in self])
    def min(self): return min([i.min() for i in self])
    def unique(self, sorted=True, return_inverse=False, return_counts=False):
        """Returns the unique elements of this tensor list.

        Args:
            sorted (bool, optional): _description_. Defaults to True.
            return_inverse (bool, optional): _description_. Defaults to False.
            return_counts (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        return torch.cat([i.ravel() for i in self]).unique(sorted=sorted, return_inverse=return_inverse, return_counts=return_counts)

    def softsign(self): return TensorList(softsign(self, foreach=self.foreach), foreach=self.foreach)

    def clamp_min(self, other: ScalarOrAnySequence): return TensorList(clamp_min(self, other, foreach=self.foreach), foreach=self.foreach)
    def clamp_min_(self, other: ScalarOrAnySequence): clamp_min_(self, other, foreach=self.foreach)
    def clamp_max(self, other: ScalarOrAnySequence): return TensorList(clamp_max(self, other, foreach=self.foreach), foreach=self.foreach)
    def clamp_max_(self, other: ScalarOrAnySequence): clamp_max_(self, other, foreach=self.foreach)


    def fastrand_like(self): return TensorList(rand_like(self, foreach=self.foreach), foreach=self.foreach)
    def fastrandn_like(self): return TensorList(randn_like(self, foreach=self.foreach), foreach=self.foreach)
    def fastrademacher_like(self): return TensorList(rademacher_like(self, foreach=self.foreach), foreach=self.foreach)
    def fastfn_like(self, fn:Callable): return TensorList(fn_like(self, fn, foreach=self.foreach), foreach=self.foreach)


    @classmethod
    def from_zeroes_like(cls, other:"TensorList"):
        return cls([torch.zeros_like(t) for t in other], foreach=other.foreach)