import numpy as np
import torch
import helper_functions as hfuncs
import time
from matplotlib import pyplot as plt


def _plot(x_t_pairs_):
    plt.scatter(x_t_pairs_[:, 0], x_t_pairs_[:, 1])
    plt.title('polynomial + Gaussian noise for m=2')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.savefig('plots/scatter.jpg')
    plt.show()


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
    train_20_y = hfuncs.polynomial_fun(w, train_20_x)  # This is the "true" polynomial curve  (i.e. "ground-truth")
    train_20_t = torch.randn_like(train_20_y) * 0.5 + train_20_y  # Add Gaussian noise to simulate the observed data.
    # train_20_t = np.random.normal(loc=train_20_y, scale=0.5)

    test_10_y = hfuncs.polynomial_fun(w, test_10_x)  # This is the "true" polynomial curve (i.e. "ground-truth")
    test_10_t = torch.randn_like(test_10_y) * 0.5 + test_10_y  # Add Gaussian noise to simulate the observed data.
    # test_10_t = np.random.normal(loc=test_10_y, scale=0.5)

    # Compute optimum weight vector:
    # x_t_pairs = np.concatenate((train_20_x, train_20_t), axis=1)
    x_t_pairs = torch.cat((train_20_x, train_20_t), dim=1)

    x_t_pairs_test = torch.cat((test_10_x, test_10_t), dim=1)

    # _plot(x_t_pairs)

    y_preds_train234 = list()
    y_preds_test234 = list()

    for i in [2, 3, 4]:
        start = time.time()
        # w_hat = hfuncs.fit_polynomial_ls(x_t_pairs=x_t_pairs, m=i)
        # print(f'fit_polynomial_ls() took {time.time() - start} secs for m={i}')
        # y_preds_train = hfuncs.polynomial_fun(w=w_hat, x=train_20_x)
        # y_preds_test = hfuncs.polynomial_fun(w=w_hat, x=test_10_x)
        #
        # # a) Print mean & standard deviation in difference between the observed training data
        # # and the underlying “true” polynomial curve.
        # # I interpret the observed to be the target, and the "true" curve to the polynomial curve.
        # diff_train = train_20_t - train_20_y
        # mean_obs_train = diff_train.mean()
        # std_obs_train = diff_train.std()
        # print(f"(Training set) Mean and standard deviation in difference between the observed training data and "
        #       f"the underlying 'true' polynomial curve is {mean_obs_train} +/- {std_obs_train}")
        # diff_test = test_10_t - test_10_y
        # mean_obs_test = diff_test.mean()
        # std_obs_test = diff_test.std()
        # print(f"(Test set) Mean and standard deviation in difference between the observed training data and "
        #       f"the underlying 'true' polynomial curve for the test set is {mean_obs_test} +/- {std_obs_test}")
        #
        # # b) Print mean & standard deviation in difference between the “LS-predicted” values
        # # and the underlying “true” polynomial curve.
        # diff_ls_train = y_preds_train - train_20_y
        # mean_ls_train = diff_ls_train.mean()
        # std_ls_train = diff_ls_train.std()
        # print(f"(Training set) Mean and standard deviation in difference between the 'LS-predicted' values and "
        #       f"the underlying 'true' polynomial curve is {mean_ls_train} +/- {std_ls_train}")
        #
        # diff_ls_test = y_preds_test - test_10_y
        # mean_ls_test = diff_ls_test.mean()
        # std_ls_test = diff_ls_test.std()
        # print(f"(Test set) Mean and standard deviation in difference between the 'LS-predicted' values and "
        #       f"the underlying 'true' polynomial curve is {mean_ls_test} +/- {std_ls_test}")

        # w_hat = hfuncs.fit_polynomial_sgd(x_t_pairs=x_t_pairs, m=i)
        w_hat = hfuncs.fit_polynomial_sgd_and_test(x_t_test_set_pairs=x_t_pairs_test, x_t_pairs=x_t_pairs, m=i)
        print(w_hat)

