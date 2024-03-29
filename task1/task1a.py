"""
Implement a task script “task1a.py”, under folder “task1”. [10]
- Experiment how to make 𝑀 a learnable model parameter and using SGD to optimise this more
flexible model.
- Report, using printed messages, the optimised 𝑀 value and the mean (and standard deviation) in
difference between the model-predicted values and the underlying “true” polynomial curve.
"""

import numpy as np
import torch
import helper_functions as hfuncs
import time
from matplotlib import pyplot as plt


def _plot(x_t_pairs_, m):
    plt.scatter(x_t_pairs_[:, 0], x_t_pairs_[:, 1])
    plt.title(f'polynomial + Gaussian noise for m={m}')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.savefig('plots/observed.jpg')
    plt.show()

# THIS IS ABLE TO RUN ON CPU OR GPU. IT SETS THE DEVICE AUTOMATICALLY.

if __name__ == '__main__':
    device = hfuncs.set_device()
    w_true = np.array([[1], [2], [3]])
    w_true = hfuncs.convert_to_tensor(w_true)

    # # Generate training set:
    x_train = np.random.uniform(low=-20, high=20, size=20).reshape(-1, 1)
    x_train = hfuncs.convert_to_tensor(x_train)

    # # Generate test set:
    x_test = np.random.uniform(low=-20, high=20, size=10).reshape(-1, 1)
    x_test = hfuncs.convert_to_tensor(x_test)

    # # "true" polynomial curve for training set
    y_true_train = hfuncs.polynomial_fun(w=w_true, x=x_train)
    t_observed_train = torch.randn_like(y_true_train) * 0.5 + y_true_train  # Add Gauss noise to simulate observed data.

    # # "true" polynomial curve for test set
    y_true_test = hfuncs.polynomial_fun(w=w_true, x=x_test)  #
    t_observed_test = torch.randn_like(y_true_test) * 0.5 + y_true_test  # Add Gauss noise to simulate observed data.

    # # Compute optimum weight vector:
    x_t_pairs = torch.cat((x_train, t_observed_train), dim=1)
    x_t_pairs_test = torch.cat((x_test, t_observed_test), dim=1)  # (only need this for the additional function.)

    # _plot(x_t_pairs)

    # It's not clear to me that the following is more "flexible", other than simply looking a very long list of m
    # values, but it's also not clear to me how else to find an optimal m, which I think can only be a discrete integer.
    lowest_rmse = 1000
    optimal_m = 0
    for m in range(10):
        # # FIT MODEL ON TRAIN SET
        w_hat_sgd = hfuncs.fit_polynomial_sgd(x_t_pairs=x_t_pairs, m=m)
        # # EVALUATE ON TEST SET
        y_preds_sgd_test = hfuncs.polynomial_fun(w=w_hat_sgd, x=x_test)
        rmse_sgd_y = hfuncs.rmse(w_or_y_true=y_true_test, w_or_y_pred=y_preds_sgd_test)
        print(f'Model fit with SGD and m={m} has RMSE for predicted y values = {rmse_sgd_y }')

        if rmse_sgd_y > lowest_rmse:
            print(f'rmse {rmse_sgd_y} for m={m} is not lowest...')
            continue
        else:
            print(f'rmse {rmse_sgd_y} for m={m} is new lowest...')
            optimal_m = m
            # OPTIMAL M THUS FAR
            lowest_rmse = rmse_sgd_y
            w_hat_sgd = hfuncs.fit_polynomial_sgd(x_t_pairs=x_t_pairs, m=m)
            y_preds_sgd_test = hfuncs.polynomial_fun(w=w_hat_sgd, x=x_test)
            diff_sgd_test = y_preds_sgd_test - y_true_test
            mean_sgd_test = hfuncs.round_sig_figs(to_round=float(diff_sgd_test.mean()), sf=4)
            std_sgd_test = hfuncs.round_sig_figs(to_round=float(diff_sgd_test.std()), sf=4)

    print(f"The optimal m is {optimal_m}. \nFor this m, the mean & std dev of difference between 'SGD-predicted' values"
              f" and underlying 'true' polynomial curve for test set = {mean_sgd_test} +/- {std_sgd_test}")



