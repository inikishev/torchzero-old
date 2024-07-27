# pylint:disable = E1102
import torch, numpy as np
@torch.no_grad
def fit_quadratic(x, y, driver=None):
    X = torch.stack([x**2, x, torch.ones_like(x)]).T
    W = torch.linalg.lstsq(X, y, driver=driver).solution
    def quadratic(x):
        X = torch.stack([x**2, x, torch.ones_like(x)]).T
        return X @ W
    return quadratic

@torch.no_grad
def find_quadratic_coefficients(x, y):
    """Returns a tensor with a, b and c in `f(x) = ax^2 + bx + c`"""
    X = torch.stack([x**2, x, torch.ones_like(x)]).T
    W = torch.linalg.lstsq(X, y).solution
    return W

@torch.no_grad
def find_minimum_x_quadratic(x, y):
    X = torch.stack([x**2, x, torch.ones_like(x)]).T
    W = torch.linalg.lstsq(X, y).solution # a, b, c
    return - W[1] / (2 * W[0])

@torch.no_grad
def fit_cubic(x, y, driver=None):
    X = torch.stack([x**3, x**2, x, torch.ones_like(x)]).T
    W = torch.linalg.lstsq(X, y, driver=driver).solution
    def cubic(x):
        X = torch.stack([x**3, x**2, x, torch.ones_like(x)]).T
        return X @ W
    return cubic

@torch.no_grad
def fit_polynomial(x, y, degree, rcond=None, driver=None):
    X = torch.stack([torch.ones_like(x), x] + [x**i for i in range(2, degree+1)]).T
    W = torch.linalg.lstsq(X, y, rcond = rcond, driver=driver).solution
    def polynomial(x):
        X = torch.stack([torch.ones_like(x), x] + [x**i for i in range(2, degree+1)]).T
        return X @ W
    return polynomial


def fit_polynomial_numpy(x, y, degree):
    """Way higher precision"""
    polynomial = np.polynomial.Polynomial.fit(x, y, degree)
    return polynomial.convert()

def find_minimum_x_quadratic_numpy(x, y):
    coeffs = np.polynomial.Polynomial.fit(x, y, 2).convert().coef
    return - coeffs[1] / (2 * coeffs[2])