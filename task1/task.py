import numpy as np
import torch
import helper_functions as hfuncs


if __name__ == '__main__':

    w = np.array([[1], [2], [3]])
    w = hfuncs.convert_to_tensor(w)

    # Generate training set:
    train_20_x = np.random.uniform(low=-20, high=20, size=20).reshape(-1, 1)
    train_20_x = hfuncs.convert_to_tensor(train_20_x)

    # Generate test set:
    test_10_x = np.random.uniform(low=-20, high=20, size=10).reshape(-1, 1)
    test_10_x = hfuncs.convert_to_tensor(test_10_x)

    # Generate observed ("target") values, using polynomial function and Gaussian noise:
    train_20_y = hfuncs.polynomial_fun(w, train_20_x)
    train_20_t = torch.randn_like(train_20_y) * 0.5 + train_20_y
    # train_20_t = np.random.normal(loc=train_20_y, scale=0.5)

    test_10_y = hfuncs.polynomial_fun(w, test_10_x)
    test_10_t = torch.randn_like(test_10_y) * 0.5 + test_10_y
    # test_10_t = np.random.normal(loc=test_10_y, scale=0.5)

    # Compute optimum weight vector:
    # x_t_pairs = np.concatenate((train_20_x, train_20_t), axis=1)
    x_t_pairs = torch.cat((train_20_x, train_20_t), dim=1)

    w_hats = list()

    for i in [2, 3, 4]:
        w_hat = hfuncs.fit_polynomial_ls(x_t_pairs=x_t_pairs, m=i)
        y_preds_train = hfuncs.polynomial_fun(w=w_hat, x=train_20_x)
        y_preds_test = hfuncs.polynomial_fun(w=w_hat, x=test_10_x)

        w_hats.append(w_hat)

    pass