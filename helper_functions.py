import numpy as np
import torch


def convert_to_tensor(a):
    """
    Convert NumPy vector or scalar to torch tensor, (and dtype float32).
    If already torch Tensor, cast to tensor of float32.
    :param a: Python integer, float or NumPy array to be cast to tensor of float32.
    :return: Torch tensor of float32 datatype.
    """
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
        a = a.to(dtype=torch.float32)
    elif isinstance(a, int) or isinstance(a, float):
        a = torch.tensor(a, dtype=torch.float32)
    elif isinstance(a, torch.Tensor):
        a = a.to(dtype=torch.float32)
    else:
        raise TypeError(f'Expected argument to be an int, float, NumPy array '
                        f'or torch.Tensor, but got {type(a)} instead.')
    return a


def polynomial_fun(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Implement polynomial function for given weight vector and scalar variable.
    :param w: Weights. A NumPy array. Shape is (m + 1, 1).
    These weights serve as the coefficients for the polynomial function for x^0, x^1, ... x^m.
    :param x: Scalar variable(s). Single scalar or NumPy array of scalars.
    :return: Function value, `y`. Column vector with same shape as `x`.
    """
    w = convert_to_tensor(w)
    x = convert_to_tensor(x)

    w = torch.reshape(w, (-1, 1))
    m = w.shape[0] - 1

    x = torch.reshape(x, (-1, 1))

    exponents = torch.arange(m + 1)
    polynomial_basis = torch.pow(x, exponents)
    y = polynomial_basis @ w
    return y


def fit_polynomial_ls(x_t_pairs: torch.Tensor, m: int) -> torch.Tensor:
    """
    Implement least squares solver for fitting polynomial functions.
    :param x_t_pairs: Pairs of predictor scalar values and corresponding target values. Torch tensor.
    :param m: Degree of polynomial to use.
    :return: Optimum weight vector, w_hat.
    """
    x = x_t_pairs[:, 0]
    t = x_t_pairs[:, 1]
    X = x.unsqueeze(dim=1)
    exponents = torch.arange(m + 1, dtype=x_t_pairs.dtype, device=x_t_pairs.device)
    exponents = exponents.unsqueeze(dim=0)
    X = torch.pow(X, exponents)
    w_hat = torch.linalg.lstsq(X, t).solution
    return w_hat



def fit_polynomial_sgd():
    pass


