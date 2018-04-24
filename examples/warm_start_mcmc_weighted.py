
# %% 

from fastFM.datasets import make_user_item_regression
from fastFM import mcmc, mcmc_weighted
from sklearn.metrics import mean_squared_error
import scipy.sparse as sp
import numpy as np


if __name__ == "__main__":


    X, y, coef = make_user_item_regression(label_stdev=.4)
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    X_train = sp.csc_matrix(X_train)
    X_test = sp.csc_matrix(X_test)

    n_iter = 50
    rank = 4
    seed = 333
    step_size = 10
    l2_reg_w = 0
    l2_reg_V = 0
    
    
    # boostrap
    
    
    ind = np.random.choice(np.arange(len(y_train)), len(y_train))
    #ind = np.arange(len(y_train))
    
    X_train_bs = X_train[ind,:]
    y_train_bs = y_train[ind]
    
    C = np.bincount(ind).astype(float)
    #C = np.ones(len(y_train))
    
    

    fm = mcmc.FMRegression(n_iter=0, rank=rank, random_state=seed)
    fm_weighted = mcmc_weighted.FMRegression_weighted(n_iter=0, rank=rank, random_state=seed)
    # initalize coefs
    fm.fit_predict(X_train_bs, y_train_bs, X_test)
    fm_weighted.fit_predict(X_train, y_train, X_test, C)

    rmse_test = []
    rmse_new = []
    rmse_test_weighted = []
    rmse_new_weighted = []
    

    
    hyper_param = np.zeros((n_iter -1, 3 + 2 * rank), dtype=np.float64)
    hyper_param_wtd = np.zeros((n_iter -1, 3 + 2 * rank), dtype=np.float64)

    for nr, i in enumerate(range(1, n_iter)):
        fm.random_state = i * seed
        y_pred = fm.fit_predict(X_train_bs, y_train_bs, X_test, n_more_iter=step_size)
        rmse_test.append(np.sqrt(mean_squared_error(y_pred, y_test)))
        hyper_param[nr, :] = fm.hyper_param_
        
        fm_weighted.random_state = i * seed +1
        y_pred_weighted = fm_weighted.fit_predict(X_train, y_train, X_test, C, n_more_iter=step_size)
        rmse_test_weighted.append(np.sqrt(mean_squared_error(y_pred_weighted, y_test)))
        hyper_param_wtd[nr, :] = fm_weighted.hyper_param_
        
  
    print ('------- restart ----------')
    values = np.arange(1, n_iter)
    rmse_test_re = []
    hyper_param_re = np.zeros((len(values), 3 + 2 * rank), dtype=np.float64)
    for nr, i in enumerate(values):
        fm = mcmc.FMRegression(n_iter=i, rank=rank, random_state=seed)
        y_pred = fm.fit_predict(X_train_bs, y_train_bs, X_test)
        rmse_test_re.append(np.sqrt(mean_squared_error(y_pred, y_test)))
        hyper_param_re[nr, :] = fm.hyper_param_
        

    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(15, 8))

    x = values * step_size
    burn_in = 5
    x = x[burn_in:]

    axes[0, 0].plot(x, rmse_test[burn_in:], label='test rmse', color="r")
    axes[0, 0].plot(values[burn_in:], rmse_test_re[burn_in:], ls="--", color="r")
    axes[0, 0].legend()

    axes[0, 1].plot(x, hyper_param[burn_in:,0], label='alpha', color="b")
    axes[0, 1].plot(values[burn_in:], hyper_param_re[burn_in:,0], ls="--", color="b")
    axes[0, 1].legend()

    axes[1, 0].plot(x, np.log(hyper_param[burn_in:,1]), label='log lambda_w', color="g")
    axes[1, 0].plot(values[burn_in:], np.log(hyper_param_re[burn_in:,1]), ls="--", color="g")
    axes[1, 0].legend()

    axes[1, 1].plot(x, hyper_param[burn_in:,3], label='mu_w', color="g")
    axes[1, 1].plot(values[burn_in:], hyper_param_re[burn_in:,3], ls="--", color="g")
    axes[1, 1].legend()

    plt.show()

    # weighted
    values = np.arange(1, n_iter)
    rmse_test_re_weighted = []
    hyper_param_re_wtd = np.zeros((len(values), 3 + 2 * rank), dtype=np.float64)
    for nr, i in enumerate(values):
        fm_weighted = mcmc_weighted.FMRegression_weighted(n_iter=i, rank=rank, random_state=seed +1)
        y_pred_weighted = fm_weighted.fit_predict(X_train, y_train, X_test, C)
        rmse_test_re_weighted.append(np.sqrt(mean_squared_error(y_pred_weighted, y_test)))
        hyper_param_re_wtd[nr, :] = fm_weighted.hyper_param_
        
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(15, 8))

    x = values * step_size
    burn_in = 5
    x = x[burn_in:]

    axes[0, 0].plot(x, rmse_test_weighted[burn_in:], label='test rmse', color="r")
    axes[0, 0].plot(values[burn_in:], rmse_test_re_weighted[burn_in:], ls="--", color="r")
    axes[0, 0].legend()

    axes[0, 1].plot(x, hyper_param_wtd[burn_in:,0], label='alpha', color="b")
    axes[0, 1].plot(values[burn_in:], hyper_param_re_wtd[burn_in:,0], ls="--", color="b")
    axes[0, 1].legend()

    axes[1, 0].plot(x, np.log(hyper_param_wtd[burn_in:,1]), label='log lambda_w', color="g")
    axes[1, 0].plot(values[burn_in:], np.log(hyper_param_re_wtd[burn_in:,1]), ls="--", color="g")
    axes[1, 0].legend()

    axes[1, 1].plot(x, hyper_param_wtd[burn_in:,3], label='mu_w', color="g")
    axes[1, 1].plot(values[burn_in:], hyper_param_re_wtd[burn_in:,3], ls="--", color="g")
    axes[1, 1].legend()

    print("OK 4")
    