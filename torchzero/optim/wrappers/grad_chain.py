from typing import Optional
from collections.abc import Callable, Sequence
import torch
from torch.optim import Optimizer

from .utils import foreach_param, foreach_group_param

class UpdateToGrad(Optimizer):
    def __init__(self, params, opt, lr:float = 1, mode: str | Callable = "closure", grad_mode: str | Callable = "set", reset=True):
        """Wraps an optimizer and converts its update into gradients that another optimizer can use.
        Update is given the different in model parameters before and after the optimizer step.

        For a more convenient way of gradient chaining you can use the `GradChain` class, which uses this internally.

        Args:
            params: same parameters that you specify for `opt`, usually `model.parameters()`.

            opt (torch.optim.Optimizer): the wrapped optimizer, e.g. `torch.optim.SGD(model.parameters(), lr=1)`

            lr (int): multiplies update by this value before converting it to gradients. For most optimizers this is the same as multiplying their learning rate. Defaults to 1. Supports parameter groups.

            mode (str | Callable): Determines how to call `optimizer.step()`. By default it does `optimizer.step(closure=closure)`. If the optimizer doesn't accept closure as an argument, set mode to `step`. It can also be a custom callable function in case the optimizer needs something different. Defaults to "closure".

            grad_mode (str | Callable): If "set", update replaces anything in `grad` attribute, if `add`, it is added to `grad` attribute, where there are probably vanilla gradients already. Supports parameter groups.

            reset (bool): If `True`, after wrapped optimizer steps and the update is converted into gradients, that update is undone. Supports parameter groups.
        """
        defaults = dict(
            lr=lr,
            grad_mode=grad_mode,
            reset=reset,
        )
        self.opt = opt
        self.mode = mode
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure:Optional[Callable] = None):
        # save parameters
        for p in foreach_param(self.param_groups):
            state = self.state[p]
            state['before'] = p.clone()

        # do a step
        if self.mode == "step": loss = self.opt.step()
        elif self.mode == "closure": loss = self.opt.step(closure)
        elif callable(self.mode): loss = self.mode(self.opt, closure, self)
        else: raise ValueError(f"Unknown mode {self.mode}")

        # convert update into gradient
        for group, p in foreach_group_param(self.param_groups):
            state = self.state[p]

            grad_mode = group["grad_mode"]
            lr = group["lr"]
            # set grad to anti-update if None or "set"
            if p.grad is None: p.grad = (state["before"] - p) * lr
            elif grad_mode == "set": p.grad.set_((state["before"] - p) * lr)
            # else add anti-update to grad
            elif grad_mode == "add": p.grad.add_((state["before"] - p) * lr)
            elif callable(grad_mode): grad_mode(p, state["before"], lr)
            else: raise ValueError(f"Unknown grad_mode {grad_mode}")

            # reset params to before update
            if group["reset"]: p.set_(state['before'])

        return loss

class GradChain(Optimizer):
    def __init__(
        self,
        params,
        optimizers: Sequence,
        lrs: float | Sequence[float] = 1,
        mode: str | Callable | Sequence[str | Callable] = "closure",
        grad_mode: str | Callable | Sequence[str | Callable] = "set",
        reset: bool | Sequence[bool] = True,
    ):
        """Gradient chaining means that after one optimizer updates parameters of the model,
        the update is undone and used as gradients for the next optimizer.

        Some uses:
        1. You can combine multiple optimizers.
        2. You can add any kind of momentum before or after any optimizer update rules by chaining it with SGD that implements momentum of your choice.
        3. You can use derivative-free optimizers to generate gradients for gradient-based optimizers.

        Args:
            params: same parameters that you specify for all optimizers in `optimizers`, usually `model.parameters()`.

            optimizers (torch.optim.Optimizer): Sequence of optimizers to chain that optimize same parameters as `params`.

            lrs: A single value or sequence of values per each optimizer. Multiplies update by this value before converting it to gradients. For most optimizers this is the same as multiplying their learning rate. Defaults to 1.

            mode: A single value or sequence of values per each optimizer. Determines how to call `optimizer.step()`. By default it does `optimizer.step(closure=closure)`. If the optimizer doesn't accept closure as an argument, set mode to `step`. It can also be a custom callable function in case the optimizer needs something different. Defaults to "closure". If you want to modify lrs for individual parameter groups, use `lr` key.

            grad_mode: A single value or sequence of values per each optimizer. If `"set"`, update replaces anything in `grad` attribute, if `"add"`, it is added to `grad` attribute, where there are probably vanilla gradients already. Supports parameter groups.

            reset: A single value or sequence of values per each optimizer. If `True`, after wrapped optimizer steps and the update is converted into gradients, that update is undone. Supports parameter groups.

        To add momentum to an optimizer, chain it with SGD that implements that momentum. For example pytorch SGD has nesterov momentum:
        ```py
        # add Nesterov momentum before optimizer update rules kick in:
        optimizer = GradChain(
            model.parameters(),
            [
                torch.optim.SGD(model.parameters(), lr=1 momentum=0.9, nesterov=True),
                MyOptimizer(model.parameters(), lr=1e-2),
            ])

        # or add Nesterov momentum after before optimizer update rules kick in:
        optimizer = GradChain(
            model.parameters(),
            [
                MyOptimizer(model.parameters(), lr=1e-2),
                torch.optim.SGD(model.parameters(), lr=1, momentum=0.9, nesterov=True),
            ])
        ```
        You can also use derivative-free optimizers to generate gradients for gradient-based optimizers:
        ```python
        optimizer = GradChain(
            model.parameters(),
            [
                torchzero.optim.RandomWalk(model.parameters(), lr=1),
                torch.optim.AdamW(model.parameters(), lr=1e-2),
            ]) # this is essentially what `RandomGrad` does.
        ```
        """
        if isinstance(lrs, (int, float)): lrs = [lrs] * (len(optimizers) - 1)
        if isinstance(mode, (str, Callable)): mode = [mode] * len(optimizers)
        if isinstance(grad_mode, (str, Callable)): grad_mode = [grad_mode] * (len(optimizers) - 1)
        if isinstance(reset, bool): reset = [reset] * (len(optimizers) - 1)

        params = list(params) # to avoid iterating over it with 1st optimizer
        self.chain = [UpdateToGrad(
                params=params,
                lr=lr,
                opt=opt,
                mode=m,
                grad_mode=gm,
                reset=r,
                )
            for lr, opt, m, gm, r in zip(lrs, optimizers[:-1], mode[:-1], grad_mode, reset)]

        self.final_optimizer = optimizers[-1]
        self.final_mode = mode[-1]

        super().__init__(params, {})

    @torch.no_grad
    def step(self, closure:Optional[Callable] = None):
        # do step through UpdateToGrad chain
        for opt in self.chain: opt.step(closure)

        # do final optimizer step
        if self.final_mode == "step": loss = self.final_optimizer.step()
        elif self.final_mode == "closure": loss = self.final_optimizer.step(closure)
        elif callable(self.final_mode): loss = self.final_mode(self.final_optimizer, closure)
        else: raise ValueError(f"Unknown mode {self.final_mode}")

        return loss # type:ignore