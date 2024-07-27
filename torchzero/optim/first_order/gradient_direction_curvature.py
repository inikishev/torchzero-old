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
        validate_step = False,
        foreach=True,
        debug=False,
    ):
        """Quadratic gradient direction curvature optimizer. You have to pass a callable closure that accepts `backward` boolean argument, for example:
        ```py
        def closure(backward: bool):
            preds = model(inputs)
            loss = loss_fn(preds, targets)
            if backward:
                optimizer.zero_grad()
                loss.backward()
            return loss

        optimizer.step(closure)

        ```

        This does two forward passes and one backward per step, obtaining three points.
        Those points are then used to fit a second order polynomial,
        which represents directional curvature in the direction of antigradient.
        If curvature is positive and not too small, a step is made to the minimum of the polynomial.
        Otherwise this acts like SGD.

        Args:
            params (_type_): _description_
            lr (_type_): This functions as both learning rate and distance to evaluate the loss again for polynomial fitting.
            lr_mul (float, optional): Epsilon for approximating loss very close the initial point using gradients will be `lr * lr_mul`. Defaults to 0.01.
            max_dist (float, optional): Clips the distance this can step towards polynomial minimum. Defaults to 8.0.
            discard_over (float, optional): Discards polynomial if distance to minimum is larger than this, to avoid instability with very small curvatures. Defaults to 8.0.
            distance_mul (float, optional): Multiplies the magnitude step towards polynomial minimum. Defaults to 1.0.
            validate_step (bool, optional): Does additional forward pass after stepping to minimum and check if loss actually decreased, if it didn't, undoes the step (so only SGD step remains). Defaults to False.
            foreach (bool, optional): Makes it faster. Defaults to True.
            debug (bool, optional): Log distances to minimum and validated loss if it is enabled. Defaults to False.
        """
        super().__init__(params, {})
        self.foreach = foreach
        self.max_dist = max_dist
        self.discard_over = discard_over
        self.lr_mul = lr_mul
        self.lr = lr
        self.distance_mul = distance_mul

        self.validate_step = validate_step

        self.debug = debug
        self.log = {}
        self.current_step = 0


    @torch.no_grad
    def step(self, closure): # type:ignore # pylint:disable=W0222
        eps = self.lr * self.lr_mul

        with torch.enable_grad(): y:torch.Tensor = closure(backward=True).detach().cpu()

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
        y_step = closure(backward=False).detach().cpu()

        if self.debug:
            log = self.log[self.current_step] = {}
            log['y'] = float(y)
            log['y_epsilon'] = float(y_epsilon)
            log['y_step'] = float(y_step)

        # make sure curvature is positive in the direction of gradient
        if y_step < y and (y - y_epsilon) / eps > (y - y_step) / self.lr:
            # find distance to minimum
            minimum_x_dist = find_minimum_x_quadratic_numpy([0, eps, self.lr], [y, y_epsilon, y_step])
            if self.debug: log['minimum_x_dist'] = float(minimum_x_dist) # type:ignore

            # make sure we don't step back
            minimum_x_dist *= self.distance_mul
            if minimum_x_dist > self.lr and minimum_x_dist < self.discard_over:
                if minimum_x_dist > self.max_dist: minimum_x_dist = self.max_dist
                if self.debug: log['step_to_minimum'] = float(minimum_x_dist) # type:ignore

                # make a step towards 0
                all_params.sub_(all_grads, alpha = minimum_x_dist - self.lr)

                # if validate_step is enabled, make sure loss didn't increase
                if self.validate_step:
                    y_validation = closure(backward=False).detach().cpu()
                    if self.debug: log['y_validation'] = float(y_validation) # type:ignore
                    # if it increased, undo the step
                    if y_validation > y_step:
                        all_params.add_(all_grads, alpha = minimum_x_dist - self.lr)

        self.current_step += 1
        return y_step