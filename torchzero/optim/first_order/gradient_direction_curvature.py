from collections.abc import Callable
import torch
import numpy as np
from torch import optim
from ..utils import get_group_params_and_grads, get_group_params_and_grads_tensorlist
from .. import _foreach
__all__ = [
    #"GD",
    "QuadraticGDC",
]

@torch.no_grad
def find_minimum_x_quadratic_numpy(x, y):
    coeffs = np.polynomial.Polynomial.fit(x, y, 2).convert().coef
    return - coeffs[1] / (2 * coeffs[2])

class QuadraticGDC(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        lr_mul=0.01,
        max_dist=8.0,
        discard_over=8.0,
        distance_mul=1.0,
        foreach=True,
        log_minimum_x=False,
    ):
        """Quadratic gradient direction curvature optimizer.

        This does two forward passes and one backward per step, obtaining three points.
        Those points are then used to fit a second order polynomial,
        which represents directional curvature in the direction of antigradient.
        If curvature is positive and not too small, a step is made to the minimum of the polynomial.
        Otherwise this does one SGD-like step.

        Args:
            params (_type_): _description_
            lr (_type_): This functions as both learning rate and distance to evaluate the loss again for polynomial fitting.
            lr_mul (float, optional): Epsilon for approximating loss very close the initial point using gradients will be `lr * lr_mul`. Defaults to 0.01.
            max_dist (float, optional): Clips the distance this can step towards polynomial minimum. Defaults to 8.0.
            discard_over (float, optional): Discards polynomial if distance to minimum is larger than this, to avoid instability with very small curvatures. Defaults to 8.0.
            distance_mul (float, optional): Multiplies the magnitude step towards polynomial minimum. Defaults to 1.0.
            foreach (bool, optional): Makes it faster. Defaults to True.
            log_minimum_x (bool, optional): Log distances to minimum to `log` attribute. Defaults to False.
        """
        super().__init__(params, {})
        self.foreach = foreach
        self.max_dist = max_dist
        self.discard_over = discard_over
        self.lr_mul = lr_mul
        self.lr = lr
        self.distance_mul = distance_mul

        self.log_minimum_x = log_minimum_x
        self.log = []


    @torch.no_grad
    def step(self, closure): # type:ignore # pylint:disable=W0222
        eps = self.lr * self.lr_mul

        with torch.enable_grad(): y:torch.Tensor = closure(backward=True)

        all_grads = _foreach.TensorList([], foreach = self.foreach)
        all_params = _foreach.TensorList([], foreach = self.foreach)

        for group in self.param_groups:
            params, grads = get_group_params_and_grads(group, with_grad=True)
            all_params.extend(params)
            all_grads.extend(grads)

        # approximate y after making a tiny step in the direction of the gradient
        y_epsilon = y - all_grads.pow(2).sum() * eps

        # make a big step in the direction of the antigradient
        all_params.sub_(all_grads, alpha = self.lr)
        y_step = closure(backward=False)

        if (y - y_epsilon) / eps > (y - y_step) / self.lr:
            minimum_x = find_minimum_x_quadratic_numpy([0, eps, self.lr], [y, y_epsilon, y_step])
            if self.log_minimum_x: self.log.append(minimum_x)

            minimum_x *= self.distance_mul
            if minimum_x > self.lr and minimum_x < self.discard_over:
                if minimum_x > self.max_dist: minimum_x = self.max_dist
                all_params.sub_(all_grads, alpha = minimum_x - self.lr)

        return y_step