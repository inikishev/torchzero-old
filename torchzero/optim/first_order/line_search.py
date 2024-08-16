from collections.abc import Callable
import torch
import numpy as np
from torch import optim
from ..utils import get_group_params_and_grads, get_group_params_and_grads_tensorlist
from .. import _foreach
__all__ = [
    #"GD",
    "QuadraticLS",
]

@torch.no_grad
def _fit_quadratic_torch(x1:float,y1:float,y1_prime:float,x2:float,y2:float,):
    """Fits a quadratic polynomial to given points and derivative.

    Args:
      x1: x-coordinate of the first point.
      y1: y-coordinate of the first point.
      y1_prime: Derivative of the function at x1.
      x2: x-coordinate of the second point.
      y2: y-coordinate of the second point.

    Returns:
      A tuple (a, b, c) representing the coefficients of the quadratic polynomial.
    """

    A = torch.tensor([[x1**2, x1, 1], [x2**2, x2, 1], [2 * x1, 1, 0]])
    b = torch.tensor([y1, y2, y1_prime])

    # Solve the system of linear equations
    coeffs = torch.linalg.solve(A, b)  # type:ignore #pylint:disable=E1102
    return coeffs

def _fit_quadratic_numpy(x1:float,y1:float,y1_prime:float,x2:float,y2:float,):
    """May give different results"""
    A = [[x1**2, x1, 1], [x2**2, x2, 1], [2 * x1, 1, 0]]
    b = [y1, y2, y1_prime]
    coeffs = np.linalg.solve(A, b)
    return coeffs

def _fit_quadratic_scipy(x1:float,y1:float,y1_prime:float,x2:float,y2:float,):
    import scipy.linalg
    A = [[x1**2, x1, 1], [x2**2, x2, 1], [2 * x1, 1, 0]]
    b = [y1, y2, y1_prime]
    coeffs = scipy.linalg.solve(A, b)  # type:ignore #pylint:disable=E1102
    return coeffs


@torch.no_grad
def _quadratic_minima(a,b): return -b / (2 * a)

class QuadraticLS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        max_dist=None,
        discard_over=None,
        distance_mul=1.0,
        validate_step = False,
        foreach=True,
        solver = _fit_quadratic_torch,
    ):
        super().__init__(params, {})
        self.foreach = foreach
        self.max_dist = max_dist
        self.discard_over = discard_over
        self.lr = float(lr)
        self.distance_mul = distance_mul

        self.validate_step = validate_step
        self.solver = solver

        self.log = {}
        self.current_step = 0


    @torch.no_grad
    def step(self, closure): # type:ignore # pylint:disable=W0222
        # f(x1)
        with torch.enable_grad(): y1 = float(closure(backward=True).detach().cpu())

        all_grads = _foreach.TensorList([], foreach = self.foreach)
        all_params = _foreach.TensorList([], foreach = self.foreach)

        for group in self.param_groups:
            params, grads = get_group_params_and_grads_tensorlist(group, with_grad=False, foreach=self.foreach)
            all_params.extend(params)
            all_grads.extend(grads)

        # directional f'(x1)
        y1_prime = float(all_grads.pow(2).sum().pow(0.5).detach().cpu()) # type:ignore

        # make a step in the direction of the antigradient
        all_params.sub_(all_grads, alpha = self.lr)

        # f(x2)
        y2 = float(closure(backward=False).detach().cpu())

        # if gradients weren't 0
        if y1_prime != 0:
            # fit a quadratic and find xmin
            a, b, c = self.solver(x1 = 0, y1 = y1, y1_prime = -y1_prime, x2 = self.lr * y1_prime, y2 = y2)
            xmin = _quadratic_minima(a, b)

            # make sure curvature is positive
            if a > 0:
                # given that we already stepped by `grad * lr`, this is remaining distance in `grad * lr`s
                distance = (xmin / (self.lr * y1_prime)) - 1

                # reduce distance by `distance_mul`
                distance *= self.distance_mul
                # discard steps that are too big
                if self.discard_over is None or xmin < self.discard_over:
                    # limit max step size
                    if self.max_dist is not None and xmin > self.max_dist: xmin = self.max_dist

                    # make a step towards xmin, given that we already stepped by `lr`
                    all_params.sub_(all_grads, alpha = self.lr * distance)

                    # if validate_step is enabled, make sure loss didn't increase
                    if self.validate_step:
                        y_val = float(closure(backward=False).detach().cpu())
                        # if it increased, move back to y2.
                        if y_val > y2:
                            all_params.add_(all_grads, alpha = self.lr * distance)

        self.current_step += 1
        return y2

@torch.no_grad
def _fit_cubic_torch(x1:float,y1:float,y1_prime:float,x2:float,y2:float, y2_prime:float):
    """Fits a cubic polynomial to given points and derivatives.

    Args:
      x1: x-coordinate of the first point.
      y1: y-coordinate of the first point.
      y1_prime: Derivative of the function at x1.
      x2: x-coordinate of the second point.
      y2: y-coordinate of the second point.
      y2_prime: Derivative of the function at x2.

    Returns:
      A tuple (a, b, c, d) representing the coefficients of the cubic polynomial.
    """

    A = torch.tensor([[x1**3, x1**2, x1, 1], [x2**3, x2**2, x2, 1], [3 * x1 ** 2, 2 * x1, 1, 0], [3 * x2 ** 2, 2 * x2, 1, 0]])
    b = torch.tensor([y1, y2, y1_prime, y2_prime])

    # Solve the system of linear equations
    coeffs = torch.linalg.solve(A, b)  # type:ignore #pylint:disable=E1102
    return coeffs

def _cubic_minima(a,b,c,d):
    x1 = (-b + (b**2 - 3 * a * c)**0.5) / (3 * a)
    x2 = (-b - (b**2 - 3 * a * c)**0.5) / (3 * a)
    y1 = a * x1 ** 3 + b * x1 ** 2 + c * x1
    y2 = a * x2 ** 3 + b * x2 ** 2 + c * x2
    if y1 < y2: return x1
    return x2