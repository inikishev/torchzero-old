import torch
from ..utils import get_group_params_and_grads_tensorlist
from .. import _foreach

__all__ = [
    "CaputoFractionalPreconditioner",
]
class CaputoFractionalPreconditioner(torch.optim.Optimizer):
    def __init__(self, params, opt = None, order = 1.75, eps = 1e-6, foreach=True):
        """Fractional `v`-th order gradient descent as described in https://www.mdpi.com/2504-3110/7/7/500.
        This is like a preconditioner, so make a step with this before making a step with the optimizer.
        or this can also wrap any other gradient-based optimizer and makes it fractional-order, if you pass it to `opt`.

        Args:
            params (_type_): Parameters to optimize, usually `model.parameters()`.

            opt (_type_, optional): The wrapped optimizer that becomes fractional. For example, `torch.optim.SGD(model.parameters())` will become FSGD (FractionalSGD) as described in the paper. If `None`, this will only modify `grad` and won't update the parameters. Defaults to None.

            v (float, optional): The fractional order of the optimizer, in `(0, 2)` range. If `v` = 1, this becomes classical gradient descent. For mysterious reasons, this parameter is called "Cnnu.nnu" in the paper. Defaults to 1.75.

            eps (_type_, optional): Small value to avoid division by zero. Defaults to 1e-6.
        """
        defaults = dict(order=order, eps=eps)
        super().__init__(params, defaults)
        self.opt = opt
        self.foreach = foreach

    @torch.no_grad
    def step(self, closure=None): # type:ignore
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            params, grads = get_group_params_and_grads_tensorlist(group, with_grad=True, foreach=self.foreach)
            # t1 = grads ^ eps
            # t2 = exp(lgamma(2 - order))
            # grads = grads * (t1 / t2)
            grads.mul_((grads ** group['eps']) / torch.tensor(2.0 - group['order']).lgamma().exp())

        if self.opt is not None: self.opt.step(closure)
        return loss