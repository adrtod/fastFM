# Author: Immanuel Bayer
# License: BSD 3 clause


import ffm
import numpy as np
from sklearn.metrics import mean_squared_error
from .validation import (assert_all_finite, check_consistent_length,
                         check_array)
from .base import (FactorizationMachine, _validate_class_labels,
                   _check_warm_start)


def find_init_stdev(fm, X_train, y_train, X_vali=None, y_vali=None,
                    stdev_range=None, ):
    if not stdev_range:
        stdev_range = [0.1, 0.1, 0.2, 0.5, 1.0]

    if not isinstance(fm, FMRegression_weighted):
        raise Exception("only implemented for FMRegression")

    # just using a dummy here
    if X_vali is None:
        X_test = X_train[:2, :]
    else:
        X_test = X_vali

    best_init_stdev = 0
    best_mse = np.finfo(np.float64).max
    for init_stdev in stdev_range:
        fm.init_stdev = init_stdev
        y_pred_vali = fm.fit_predict(X_train, y_train, X_test)
        if X_vali is None:
            y_pred = fm.predict(X_train)
            mse = mean_squared_error(y_pred, y_train)
        else:
            mse = mean_squared_error(y_pred_vali, y_vali)
        if mse < best_mse:
            best_mse = mse
            best_init_stdev = init_stdev
    return best_init_stdev, best_mse


def _validate_mcmc_fit_input(X_train, y_train, X_test):

        check_consistent_length(X_train, y_train)
        assert_all_finite(y_train)
        y_train = check_array(y_train, ensure_2d=False, dtype=np.float64)

        assert X_train.shape[1] == X_test.shape[1]
        X_train = check_array(X_train, accept_sparse="csc", dtype=np.float64,
                              order="F")
        X_test = check_array(X_test, accept_sparse="csc", dtype=np.float64,
                             order="F")
        return X_train, y_train, X_test


class FMRegression_weighted(FactorizationMachine):
    """ Factorization Machine Regression with a MCMC solver.

    Parameters
    ----------
    n_iter : int, optional
        The number of samples for the MCMC sampler, number or iterations over
        the training set for ALS and number of steps for SGD.

    init_stdev: float, optional
        Sets the stdev  for the initialization of the parameter

    random_state: int, optional
        The seed of the pseudo random number generator that
        initializes the parameters and mcmc chain.

    rank: int
        The rank of the factorization used for the second order interactions.


    Attributes
    ----------
    w0_ : float
        bias term

    w_ : float | array, shape = (n_features)
        Coefficients for linear combination.

    V_ : float | array, shape = (rank_pair, n_features)
        Coefficients of second order factor matrix.
    """

    def fit_predict(self, X_train, y_train, X_test, C, n_more_iter=0):
        """Return average of posterior estimates of the test samples.

        Parameters
        ----------
        X_train : scipy.sparse.csc_matrix, (n_samples, n_features)

        y_train : array, shape (n_samples)

        X_test : scipy.sparse.csc_matrix, (n_test_samples, n_features)

        n_more_iter : int
                Number of iterations to continue from the current Coefficients.

        Returns
        -------
        T : array, shape (n_test_samples)
        """
        self.task = "regression"
        X_train, y_train, X_test = _validate_mcmc_fit_input(X_train, y_train,
                                                            X_test)

        self.n_iter = self.n_iter + n_more_iter

        if n_more_iter > 0:
            _check_warm_start(self, X_train)
            assert self.prediction_.shape[0] == X_test.shape[0]
            assert self.hyper_param_.shape
            self.warm_start = True
        else:
            self.iter_count = 0

        coef, y_pred = ffm.ffm_mcmc_fit_predict_weighted(self, X_train,
                                                X_test, y_train, C)
        self.w0_, self.w_, self.V_ = coef
        self.prediction_ = y_pred
        self.warm_start = False

        if self.iter_count != 0:
            self.iter_count = self.iter_count + n_more_iter
        else:
            self.iter_count = self.n_iter

        return y_pred

