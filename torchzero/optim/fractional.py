import torch

class FractionalOptimizer(torch.optim.Optimizer):
    def __init__(self, params, opt = None, v = 1.75, eps = 1e-6):
        """Fractional `v`-th order gradient descent as described in https://www.mdpi.com/2504-3110/7/7/500.
        This wraps any other gradient-based optimizer and makes it fractional-order. It modifies `grad` attribute before making a step.

        Args:
            params (_type_): Parameters to optimize, usually `model.parameters()`.

            opt (_type_, optional): The wrapped optimizer that becomes fractional. For example, `torch.optim.SGD(model.parameters())` will become FSGD (FractionalSGD) as described in the paper. If `None`, this will only modify `grad` and won't update the parameters. Defaults to None.

            v (float, optional): The fractional order of the optimizer, in `(0, 2)` range. If `v` = 1, this becomes classical gradient descent. For mysterious reasons, this parameter is called "Cnnu.nnu" in the paper. Defaults to 1.75.

            eps (_type_, optional): Small value to avoid division by zero. Defaults to 1e-6.
        """
        defaults = dict(v=v, eps=eps)
        super().__init__(params, defaults)
        self.opt = opt

    def step(self, closure=None): # type:ignore
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    v = group['v']
                    t1 = torch.pow(p.grad.abs() + group['eps'], 1 - v) # t 1 = t o r c h . pow( abs ( d_p ) +eps , 1−v )
                    t2 = torch.exp(torch.lgamma(torch.tensor(2.0 - v))) # 2 = t o r c h . exp ( t o r c h . lgamma ( t o r c h . t e n s o r ( 2 . 0 − v ) ) )
                    p.grad.mul_(t1 / t2) # d_p = d_p * t 1 / t 2

        if self.opt is not None: self.opt.step(closure)
        return loss