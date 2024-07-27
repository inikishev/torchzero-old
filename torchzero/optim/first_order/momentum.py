import torch
from torch import optim
from ..utils import get_group_params_and_grads_tensorlist
from ...random.random import randmask
from .. import _foreach

__all__ = [
    "PolyakMomentum",
]
class PolyakMomentum(optim.Optimizer):
    def __init__(self, params, lr, b=0.6, foreach=True):
        """Polyak's Momentum (Heavy Ball Method).
        Polyak, Boris T. “Some methods of speeding up the convergence of iteration methods.” USSR Computational Mathematics and Mathematical Physics 4, no. 5 (1964): 1-17.

        Args:
            params (_type_): _description_
            lr (_type_): _description_
            foreach (bool, optional): _description_. Defaults to True.
        """
        super().__init__(params, dict(lr=lr, b=b))

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['last update'] = torch.zeros_like(p)

        self.foreach = foreach

    @torch.no_grad
    def step(self, closure=None): # type:ignore
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            params, grads = get_group_params_and_grads_tensorlist(group, with_grad=True, foreach=self.foreach)
            lr = group['lr']
            b = group['b']

            last_update = _foreach.TensorList([self.state[p]['last update'] for p in params], foreach=self.foreach)
            last_update.sub_(grads, alpha = lr)

            # params = params - grads * lr
            params.add_(last_update)

            # params = params + b * (params - prev_params)
            last_update.mul_(b)

        return loss
