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
        Now we add the difference between current and previous parameters multiplied by `b`.
        Meaning if current direction is similar to previous one, we go a bit further, and if it is different, we go less far.
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
                state['w(t)'] = torch.zeros_like(p)

        self.foreach = foreach

    @torch.no_grad
    def step(self, closure=None): # type:ignore
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            params, grads = get_group_params_and_grads_tensorlist(group, with_grad=True, foreach=self.foreach)

            for p in params:
                self.state[p]['w(t-1)'] = self.state[p]['w(t)']
                self.state[p]['w(t)'] = p.clone()

            wt_1 = _foreach.TensorList([self.state[p]['w(t-1)'] for p in params], foreach=self.foreach)
            wt = _foreach.TensorList([self.state[p]['w(t)'] for p in params], foreach=self.foreach)

            # params = params - grads * lr
            params.sub_(grads, alpha=group['lr'])

            # params = params + b * (params - prev_params)
            params.add_(wt - wt_1, alpha=group['b'])

        return loss
