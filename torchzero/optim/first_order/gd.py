import torch
from torch import optim
from ..utils import get_group_params_and_grads, get_group_params_and_grads_tensorlist
from ...random.random import randmask
from .. import _foreach
__all__ = [
    #"GD",
    "SignGD",
    "BitGD",
    "SoftSignGD",
    "RootGD",
]
#region GD
class GD(optim.Optimizer):
    def __init__(self, params, lr, foreach=True):
        """Gradient descent (reference implementation, just use torch.optim.SGD).

        Args:
            params (_type_): _description_
            lr (_type_): _description_
            foreach (bool, optional): _description_. Defaults to True.
        """
        super().__init__(params, dict(lr=lr))
        self.foreach = foreach

    @torch.no_grad
    def step(self, closure=None): # type:ignore
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            params, grads = get_group_params_and_grads(group, with_grad=True)
            _foreach.sub_(params, grads, alpha = group['lr'], foreach = self.foreach)

        return loss
#endregion
#region SignGD
class SignGD(optim.Optimizer):
    def __init__(self, params, lr, foreach=True):
        """Sign gradient descent. Uses sign of the gradient to update the parameters.

        Args:
            params (_type_): _description_
            lr (_type_): _description_
            foreach (bool, optional): _description_. Defaults to True.
        """
        super().__init__(params, dict(lr=lr))
        self.foreach = foreach

    @torch.no_grad
    def step(self, closure=None): # type:ignore
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            params, grads = get_group_params_and_grads_tensorlist(group, with_grad=True, foreach=self.foreach)
            params.sub_(grads.sign(), alpha = group['lr'])

        return loss
#endregion
#region BitGD
class BitGD(optim.Optimizer):
    def __init__(self, params, lr, foreach = True):
        """Operates on parameters that can have ether 1 or -1 as the value.

        Args:
            params (_type_): _description_
            lr (_type_): Probability that a parameter will be updated by the gradient.
        """
        super().__init__(params, dict(lr=lr))

        self.foreach = foreach

    @torch.no_grad
    def step(self, closure=None): # type:ignore
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            params, grads = get_group_params_and_grads_tensorlist(group, with_grad=True, foreach=self.foreach)
            # assign new weights to params.data
            params.set_(
                # generate a random mask with `lr` probability for each value to be True;
                # where mask is True, set parameters to `- gradient.sign()`.
                [p.where(randmask(grad.size(), p = group['lr'], device=grad.device), grad) for p,grad in zip(
                    params,
                    # negate the sign of the gradient
                    grads.sign().neg()
                )]
            )
        return loss
#endregion

#region SoftSignGD
class SoftSignGD(optim.Optimizer):
    def __init__(self, params, lr, foreach = True):
        """Uses softsign of the gradient to update the parameters.

        Args:
            params (_type_): _description_
            lr (_type_): _description_
            foreach (bool, optional): _description_. Defaults to True.
        """
        super().__init__(params, dict(lr=lr))
        self.foreach = foreach

    @torch.no_grad
    def step(self, closure=None): # type:ignore
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            params, grads = get_group_params_and_grads_tensorlist(group, with_grad=True, foreach=self.foreach)
            params.sub_(grads.softsign(), alpha=group['lr'])
        return loss
#endregion

#region RootGD
class RootGD(optim.Optimizer):
    def __init__(self, params, lr, root=2., foreach = True):
        """Uses root of the gradient to update the parameters. The higher the root is, the closer this is to SignGD.

        Args:
            params (_type_): _description_
            lr (_type_): _description_
            root (_type_): _description_
            foreach (bool, optional): _description_. Defaults to True.
        """
        super().__init__(params, dict(lr=lr, root=root))
        self.foreach = foreach

    @torch.no_grad
    def step(self, closure=None): # type:ignore
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            params, grads = get_group_params_and_grads_tensorlist(group, with_grad=True, foreach=self.foreach)
            params.sub_(grads.abs().pow(1 / group['root']).mul(grads.sign()), alpha=group['lr'])
        return loss
#endregion

#region PowerGD
class PowerGD(optim.Optimizer):
    def __init__(self, params, lr, power=2, foreach = True):
        """Uses power of the gradient to update the parameters.

        This is kinda like the opposite of SignGD, and usually does worse than normal GD.

        Args:
            params (_type_): _description_
            lr (_type_): _description_
            power (_type_): _description_
            foreach (bool, optional): _description_. Defaults to True.
        """
        super().__init__(params, dict(lr=lr, power=power))
        self.foreach = foreach

    @torch.no_grad
    def step(self, closure=None): # type:ignore
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            power = group['power']
            params, grads = get_group_params_and_grads_tensorlist(group, with_grad=True, foreach=self.foreach)
            if power % 2 == 0: params.sub_(grads.pow(group['power']).mul(grads.sign()), alpha=group['lr'])
            else: params.sub_(grads.pow(group['power']), alpha=group['lr'])
        return loss
#endregion