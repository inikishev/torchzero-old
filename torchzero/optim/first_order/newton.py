import torch
from torch import optim
from ..utils import get_group_params_and_grads, get_group_params_and_grads_tensorlist
from .. import _foreach

__all__ = [
    "NewtonsMethod",
    "SemiNewton",
]
#region Newton's Method
class NewtonsMethod(optim.Optimizer):
    def __init__(self, params, lr = 1., grad_min = 0.01, foreach=True):
        """Newton's root finding method.

        Args:
            params (_type_): _description_
            lr (_type_, optional): _description_. Defaults to 1..
            foreach (bool, optional): _description_. Defaults to True.
        """
        super().__init__(params, dict(lr=lr, grad_min=grad_min))
        self.foreach = foreach

    @torch.no_grad
    def step(self, closure=None, loss=None): # type:ignore
        if loss is None:
            with torch.enable_grad(): loss = closure() # type:ignore

        for group in self.param_groups:
            params, grads = get_group_params_and_grads(group, with_grad=True)
            # params = params - (loss / grads)
            _foreach.sub_(
                params,
                _foreach.div(
                    [loss for _ in grads],
                    _foreach.clamp_min(grads, group['grad_min'], foreach=self.foreach),
                    foreach=self.foreach
                ),
                alpha = group['lr'],
                foreach = self.foreach,
            )

        return loss
#endregion

#region Semi-Newton's
class SemiNewton(optim.Optimizer):
    def __init__(self, params, lr, threshold = 'fixed', loss_multiplier = 1., fixed_value = 1., foreach=True,):
        """_summary_

        Args:
            params (_type_): _description_
            lr (_type_): _description_
            threshold (str, optional): _description_. Defaults to 'loss'.
            loss_multiplier (_type_, optional): _description_. Defaults to 1..
            fixed_value (int, optional): _description_. Defaults to 1.
            foreach (bool, optional): _description_. Defaults to True.
        """
        super().__init__(params, dict(lr=lr, threshold = threshold, fixed_value = fixed_value, loss_multiplier=loss_multiplier))
        self.foreach = foreach

    @torch.no_grad
    def step(self, closure=None, loss=None): # type:ignore
        if loss is None and closure is not None:
            with torch.enable_grad(): loss = closure() # type:ignore

        if loss == 0: return loss

        for group in self.param_groups:
            params, grads = get_group_params_and_grads(group, with_grad=True)
            threshold_mode = group['threshold']
            if threshold_mode == 'loss': threshold = loss * group['loss_multiplier']
            elif threshold_mode == 'fixed': threshold = group['fixed_value']
            else: raise ValueError(f"Invalid threshold mode: {threshold_mode}")
            # update = {grad, where grad < loss;
            #           1 / grad otherwise}
            # params = params - (loss / update)
            _foreach.sub_(
                params,
                [g.where(g<threshold, (threshold**2)/g) for g in grads], alpha = group['lr'],
                foreach = self.foreach
            )

        return loss
#endregion
