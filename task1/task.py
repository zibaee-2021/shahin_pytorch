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

    # # Generate observed ("target") values, using polynomial function and Gaussian noise:
    y_true_train = hfuncs.polynomial_fun(w=w_true, x=x_train)  # This is the "true" polynomial curve  (i.e.
    # "ground-truth")
    t_observed_train = torch.randn_like(y_true_train) * 0.5 + y_true_train  # Add Gauss noise to simulate observed data.

    y_true_test = hfuncs.polynomial_fun(w=w_true, x=x_test)  # This is the "true" polynomial curve (i.e. "ground-truth")
    t_observed_test = torch.randn_like(y_true_test) * 0.5 + y_true_test  # Add Gaussian noise to simulate observed data.

    # # Compute optimum weight vector:
    x_t_pairs = torch.cat((x_train, t_observed_train), dim=1)
    x_t_pairs_test = torch.cat((x_test, t_observed_test), dim=1)  # (only need this for the additional function.)

    # _plot(x_t_pairs)

    for m in [2, 3, 4]:

        # # OBSERVED --------------------------------------------------------------------------------------------------

        # # "Report, using printed messages, the mean (and standard deviation) in difference a) between
        # # the observed training data and the underlying “true” polynomial curve; and b) between the
        # # “LS-predicted” values and the underlying “true” polynomial curve."
        # # I interpret the "observed" to mean the "target", and the "true" polynomial curve as that generated by
        # # polynomial_fun() with degree=2 and w=[1, 2, 3].
        diff_train = t_observed_train - y_true_train
        mean_obs_train = hfuncs.round_sig_figs(to_round=float(diff_train.mean()), sf=4)
        std_obs_train = hfuncs.round_sig_figs(to_round=float(diff_train.std()), sf=4)
        print(f"For m={m}, Mean & std dev of difference between observed training data and "
              f"underlying 'true' polynomial curve for training set = {mean_obs_train} +/- {std_obs_train} (to 4 s.f.)")

        diff_test = t_observed_test - y_true_test
        mean_obs_test = hfuncs.round_sig_figs(to_round=float(diff_test.mean()), sf=4)
        std_obs_test = hfuncs.round_sig_figs(to_round=float(diff_test.std()), sf=4)
        print(f"For m={m}, Mean & std dev of difference between observed training data and "
              f"underlying 'true' polynomial curve for test set = {mean_obs_test} +/- {std_obs_test} (to 4 s.f.).")

        # # LS-PREDICTED ----------------------------------------------------------------------------------------------

        # # "report time spent in fitting/training (in seconds) using printed messages."
        start = time.time()
        w_hat_ls = hfuncs.fit_polynomial_ls(x_t_pairs=x_t_pairs, m=m)
        print(f'fit_polynomial_ls() took {round(time.time() - start, 5)} secs (to 5 d.p.), for m={m}')
        y_preds_ls_train = hfuncs.polynomial_fun(w=w_hat_ls, x=x_train)
        y_preds_ls_test = hfuncs.polynomial_fun(w=w_hat_ls, x=x_test)

        # # "Report, using printed messages, the mean (and standard deviation) in difference a) between
        # # the observed training data and the underlying “true” polynomial curve; and b) between the
        # # “LS-predicted” values and the underlying “true” polynomial curve."
        diff_ls_train = y_preds_ls_train - y_true_train
        mean_ls_train = hfuncs.round_sig_figs(to_round=float(diff_ls_train.mean()), sf=4)
        std_ls_train = hfuncs.round_sig_figs(to_round=float(diff_ls_train.std()), sf=4)
        print(f"For m={m}, Mean & std dev of difference between 'LS-predicted' values and underlying 'true' "
              f"polynomial curve for training set = {mean_ls_train} +/- {std_ls_train} (to 4 s.f.)")

        diff_ls_test = y_preds_ls_test - y_true_test
        mean_ls_test = hfuncs.round_sig_figs(to_round=float(diff_ls_test.mean()), sf=4)
        std_ls_test = hfuncs.round_sig_figs(to_round=float(diff_ls_test.std()), sf=4)
        print(f"For m={m}, Mean & std dev of difference between the 'LS-predicted' values"
              f"and underlying 'true' polynomial curve for test set = {mean_ls_test} +/- {std_ls_test} (to 4 s.f.)")

        print(f'LS-learned weights = {w_hat_ls.numpy().ravel()}')

        # # SGD-PREDICTED ---------------------------------------------------------------------------------------------

        # # "report time spent in fitting/training (in seconds) using printed messages."
        start = time.time()
        w_hat_sgd = hfuncs.fit_polynomial_sgd(x_t_pairs=x_t_pairs, m=m)
        print(f'fit_polynomial_sgd() took {round(time.time() - start, 5)} secs (to 5 d.p) for m={m}')

        # # For a more informative method, uncomment and run the following function which is identical to
        # # fit_polynomial_sgd, but includes LEARNING CURVES of the BOTH TRAINING AND TEST sets together:
        # w_hat_sgd = hfuncs.fit_polynomial_sgd_and_test(x_t_test_set_pairs=x_t_pairs_test, x_t_pairs=x_t_pairs, m=m)

        # # "Report, using printed messages, the mean (and standard deviation) in difference between the
        # # “SGD-predicted” values and the underlying “true” polynomial curve."
        y_preds_sgd_train = hfuncs.polynomial_fun(w=w_hat_sgd, x=x_train)
        y_preds_sgd_test = hfuncs.polynomial_fun(w=w_hat_sgd, x=x_test)

        diff_sgd_train = y_preds_sgd_train - y_true_train
        mean_sgd_train = hfuncs.round_sig_figs(to_round=float(diff_sgd_train.mean()), sf=4)
        std_sgd_train = hfuncs.round_sig_figs(to_round=float(diff_sgd_train.std()), sf=4)
        print(f"For m={m}, Mean & std dev of difference between 'SGD-predicted' values and underlying 'true' "
              f"polynomial curve for training set = {mean_sgd_train} +/- {std_sgd_train} (to 4 s.f.)")

        diff_sgd_test = y_preds_sgd_test - y_true_test
        mean_sgd_test = hfuncs.round_sig_figs(to_round=float(diff_sgd_test.mean()), sf=4)
        std_sgd_test = hfuncs.round_sig_figs(to_round=float(diff_sgd_test.std()), sf=4)
        print(f"For m={m}, Mean & std dev of difference between 'SGD-predicted' values"
              f"and underlying 'true' polynomial curve for test set = {mean_sgd_test} +/- {std_sgd_test}")
        print(f'SGD-learned weights = {w_hat_sgd.numpy().ravel()}')

        # # ACCURACIES -----------------------------------------------------------------------------------------------

        # # "report the root-mean-square-errors (RMSEs) in both 𝐰 and 𝑦 using printed messages LS-predicted."
        rmse_ls_w = hfuncs.rmse(w_or_y_true=w_true, w_or_y_pred=w_hat_ls)
        print(f'For m={m}, LS-predicted weights = {w_hat_ls.numpy().ravel()}, giving RMSE = {rmse_ls_w}')

        rmse_ls_y = hfuncs.rmse(w_or_y_true=y_true_test, w_or_y_pred=y_preds_ls_test)
        print(f'For m={m}, LS-predicted y values = {y_preds_ls_test.numpy().ravel()}, giving RMSE = {rmse_ls_y}')

        # # SGD-predicted:
        rmse_sgd_w = hfuncs.rmse(w_or_y_true=w_true, w_or_y_pred=w_hat_sgd)
        print(f'For m={m}, SGD-predicted weights = {w_hat_sgd.numpy().ravel()}, giving RMSE = {rmse_sgd_w}')

        rmse_sgd_y = hfuncs.rmse(w_or_y_true=y_true_test, w_or_y_pred=y_preds_sgd_test)
        print(f'For m={m}, SGD-predicted y values = {y_preds_sgd_test.numpy().ravel()}, giving RMSE = {rmse_sgd_y}')

