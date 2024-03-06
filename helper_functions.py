import numpy as np
import torch
import torch.nn as nn
import torch.nn as nn
import torch.optim as optim
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt


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
    :param w: Weights. Torch tensor of shape is (m + 1, 1).
    These weights serve as the coefficients for the polynomial function for x^0, x^1, ... x^m.
    :param x: Independent scalar variable(s) in tensor (column vector), e.g. shape (20,1) or (10,1).
    :return: Function value, `y`. Column vector with same shape as `x`.
    """
    # In case tester calls this function with other than torch.Tensors:
    w = convert_to_tensor(w)
    x = convert_to_tensor(x)
    w = torch.reshape(w, (-1, 1))
    m = w.shape[0] - 1
    x = torch.reshape(x, (-1, 1))  # In case tester calls this function separately with row vector or series
    exponents = torch.arange(m + 1, dtype=x.dtype, device=x.device)
    X = torch.pow(x, exponents)
    y = X @ w
    return y


def fit_polynomial_ls(x_t_pairs: torch.Tensor, m: int) -> torch.Tensor:
    """
    Implement least squares solver for fitting polynomial functions for given dataset, to given degree (plus bias).
    :param x_t_pairs: Pairs of predictor scalar values and corresponding target values. Torch tensor, shape (20,2).
    :param m: Degree of polynomial to use.
    :return: Optimum weight vector, w_hat.
    """
    x = x_t_pairs[:, 0]
    t = x_t_pairs[:, 1]
    x = x.unsqueeze(dim=1)  # Convert to 2d column vector (same as reshape(-1,1))
    exponents = torch.arange(m + 1, dtype=x_t_pairs.dtype, device=x_t_pairs.device)
    # exponents = exponents.unsqueeze(dim=0) # Convert to 2d row vector
    X = torch.pow(x, exponents)
    w_hat = torch.linalg.lstsq(X, t).solution
    return w_hat


def fit_polynomial_sgd(x_t_pairs: torch.Tensor, m: int, lr=0.001, mb_size=2) -> torch.Tensor:
    """
    Implement stochastic minibatch gradient descent for fitting polynomial functions for given dataset,
    to given degree (plus bias).
    :param x_t_pairs: Pairs of predictor scalar values and corresponding target values. Torch tensor, shape (20,2).
    :param m: Degree of polynomial to use. Expected to be 2, 3, or 4.
    :param lr: learning rate. A regularisation, for countering over-fitting effects.
    :param mb_size: Minibatch size. Expected to be within range 1-20, more likely to be within ~ 2-10.
    :return: Optimum weight vector, w_hat.
    """
    class SGDNet(nn.Module):
        def __init__(self, m):
            super(SGDNet, self).__init__()
            # Random init of weights
            self.weights = nn.Parameter(torch.randn(m + 1, 1))

        def forward(self, x_: torch.Tensor):
            x_ = torch.reshape(x_, (-1, 1))  # Just in case
            exponents = torch.arange(m + 1, dtype=x_.dtype, device=x_.device)
            X = torch.pow(x_, exponents)
            y_pred = X @ self.weights
            return y_pred

    x_t_pairs = convert_to_tensor(x_t_pairs)  # just in case
    x = x_t_pairs[:, 0].unsqueeze(dim=1)
    t = x_t_pairs[:, 1].unsqueeze(dim=1)

    momentum = 0.9
    model = SGDNet(m)
    optimiser = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.MSELoss()
    gc_max_norm = 1.0

    ds = TensorDataset(x, t)
    dataloader = DataLoader(dataset=ds, batch_size=mb_size, shuffle=True)
    num_epochs = 5000
    losses = list()
    loss = 0

    # Training loop:
    for epoch in range(num_epochs):

        for batch_i, (x_mb, t_mb) in enumerate(dataloader):

            optimiser.zero_grad()  # Clear gradients from last iteration.

            y_pred_mb = model(x_mb)  # Forward pass
            loss = criterion(y_pred_mb, t_mb)  # Calculate loss.

            loss.backward()  # Backward pass: compute gradient of loss with respect to weights.

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gc_max_norm)  # Gradient clipping.

            optimiser.step()  # Update weights using gradient descent.
            # print(f'Epoch [{epoch + 1} / {num_epochs}], Batch [{batch_i + 1} / {len(dataloader)}], Loss: {loss.item()}')

        losses.append(loss.item())
        # Evaluation using test set:
        with torch.no_grad():  # Ensure no gradient computation for test data
            y_test_pred = model(test_x)  # Forward pass with test data
            test_loss = criterion(y_test_pred, test_t)  # Calculate loss on test data
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Test Loss: {test_loss.item()}')

        print(f'Epoch [{epoch + 1} / {num_epochs}], Loss: {loss.item()}')
    _plot_loss(losses, num_epochs, lr=lr, momentum=momentum, mb_size=mb_size, gc_max_norm=gc_max_norm)
    return model.weights.data


def fit_polynomial_sgd_and_test(x_t_test_set_pairs: torch.Tensor, x_t_pairs: torch.Tensor, m: int,
                       lr=0.001, mb_size=2) -> torch.Tensor:
    """
    Implement stochastic minibatch gradient descent for fitting polynomial functions for given dataset,
    to given degree (plus bias).
    :param x_t_test_set_pairs: Pairs of test set values and corresponding target values. Torch tensor, shape (10,2).
    :param x_t_pairs: Pairs of predictor scalar values and corresponding target values. Torch tensor, shape (20,2).
    :param m: Degree of polynomial to use. Expected to be 2, 3, or 4.
    :param lr: learning rate. A regularisation, for countering over-fitting effects.
    :param mb_size: Minibatch size. Expected to be within range 1-20, more likely to be within ~ 2-10.
    :return: Optimum weight vector, w_hat.
    """
    class SGDNet(nn.Module):
        def __init__(self, m):
            super(SGDNet, self).__init__()
            # Random init of weights
            self.weights = nn.Parameter(torch.randn(m + 1, 1))

        def forward(self, x_: torch.Tensor):
            x_ = torch.reshape(x_, (-1, 1))  # Just in case
            exponents = torch.arange(m + 1, dtype=x_.dtype, device=x_.device)
            X = torch.pow(x_, exponents)
            y_pred = X @ self.weights
            return y_pred

    # Train set
    x_t_pairs = convert_to_tensor(x_t_pairs)  # just in case
    x = x_t_pairs[:, 0].unsqueeze(dim=1)
    t = x_t_pairs[:, 1].unsqueeze(dim=1)

    # Test set
    x_t_test_set_pairs = convert_to_tensor(x_t_test_set_pairs)  # just in case
    x_test = x_t_test_set_pairs[:, 0].unsqueeze(dim=1)
    t_test = x_t_test_set_pairs[:, 1].unsqueeze(dim=1)

    momentum = 0.9
    model = SGDNet(m)
    optimiser = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.MSELoss()
    gc_max_norm = 1.0

    ds = TensorDataset(x, t)
    dataloader = DataLoader(dataset=ds, batch_size=mb_size, shuffle=True)
    num_epochs, loss = 300, 0
    losses, test_losses = list(), list()

    # Training loop:
    for epoch in range(num_epochs):

        for batch_i, (x_mb, t_mb) in enumerate(dataloader):

            optimiser.zero_grad()  # Clear gradients from last iteration.

            y_pred_mb = model(x_mb)  # Forward pass
            loss = criterion(y_pred_mb, t_mb)  # Calculate loss.

            loss.backward()  # Backward pass: compute gradient of loss with respect to weights.

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gc_max_norm)  # Gradient clipping.

            optimiser.step()  # Update weights using gradient descent.
            # print(f'Epoch [{epoch + 1} / {num_epochs}], Batch [{batch_i + 1} / {len(dataloader)}], Loss: {loss.item()}')

        losses.append(loss.item())

        # Evaluation using test set:
        with torch.no_grad():  # Ensure no gradient computation for test data
            y_test_pred = model(x_test)  # Forward pass with test data
            test_loss = criterion(y_test_pred, t_test)  # Calculate loss on test data
        print(f'Epoch [{epoch + 1} / {num_epochs}], Loss: {loss.item()}, Test Loss: {test_loss.item()}')
        test_losses.append(test_loss.item())
    _plot_losses(test_losses, losses, num_epochs, lr=lr, momentum=momentum, mb_size=mb_size,
                     gc_max_norm=gc_max_norm)
    return model.weights.data


# A separate identical function which includes the test set argument is included for purposes of making a learning
# curve of both the test and training datasets.


def _plot_loss(losses, num_epochs: int, lr: float, momentum: float, mb_size: int, gc_max_norm: float):
    losses = np.array(losses)
    epochs = np.arange(1, num_epochs + 1)
    _, ax = plt.subplots()
    ax.set_xlim(1, num_epochs + 1)
    # ax.set_ylim(0, 100)
    plt.title(f'num_epochs={num_epochs}, lr={lr}, momentum={momentum}, mb_size={mb_size}, \n'
              f'gc_max_norm={gc_max_norm}')
    ax.scatter(epochs, losses, color='red', s=10)
    ax.plot(epochs, losses)
    plt.xlabel('epoch')
    plt.ylabel(f'Loss')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()


def _plot_losses(test_losses, losses, num_epochs: int, lr: float, momentum: float, mb_size: int, gc_max_norm: float):
    losses = np.array(losses)
    test_losses = np.array(test_losses)
    epochs = np.arange(1, num_epochs + 1)
    _, ax = plt.subplots()
    ax.set_xlim(1, num_epochs + 1)
    # ax.set_ylim(0, 100)
    plt.title(f'num_epochs={num_epochs}, lr={lr}, momentum={momentum}, mb_size={mb_size}, \n'
              f'gc_max_norm={gc_max_norm}')
    ax.scatter(epochs, losses, label='train', color='blue', s=10)
    ax.scatter(epochs, test_losses, label='test', color='red', s=10)

    ax.plot(epochs, losses, color='lightblue')
    ax.plot(epochs, test_losses, color='salmon')

    plt.xlabel('epoch')
    plt.ylabel(f'Losses')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()
    plt.show()
