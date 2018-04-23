#%%

from fastFM.datasets import make_user_item_regression
from fastFM import als, als_weighted
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

    """
    offset = '../../fastFM-notes/benchmarks/'
    train_path = offset + "data/ml-100k/u1.base.libfm"
    test_path = offset + "data/ml-100k/u1.test.libfm"

    from sklearn.datasets import load_svmlight_file
    X_train, y_train = load_svmlight_file(train_path)
    X_test,  y_test= load_svmlight_file(test_path)
    X_train = sp.csc_matrix(X_train)
    X_test = sp.csc_matrix(X_test)
    # add padding for features not in test
    X_test = sp.hstack([X_test, sp.csc_matrix((X_test.shape[0], X_train.shape[1] - X_test.shape[1]))])
    """

    n_iter = 500
    rank = 4
    seed = 333
    step_size = 10
    l2_reg_w = 0
    l2_reg_V = 0
    
    
    # boostrap
    ind = np.random.choice(np.arange(len(y_train)), len(y_train))
    
    X_train_bs = X_train[ind,:]
    y_train_bs = y_train[ind]
    
    C = np.bincount(ind).astype(float)
    

    fm = als.FMRegression(n_iter=0, l2_reg_w=l2_reg_w,
            l2_reg_V=l2_reg_V, rank=rank, random_state=seed)
    # initalize coefs
    fm.fit(X_train_bs, y_train_bs)
    
    # weighted
    fm_weighted = als_weighted.FMRegression_weighted(n_iter=0, l2_reg_w=l2_reg_w,
            l2_reg_V=l2_reg_V, rank=rank, random_state=seed)
    #C = np.ones(len(y))/len(y)
    fm_weighted.fit(X_train, y_train, C)
    
    rmse_train = []
    rmse_test = []
    
    # weighted
    rmse_train_wtd = []
    rmse_test_wtd = []
    
    values = np.arange(step_size, n_iter, step_size)
    
    for i in range(len(values)):
        fm.fit(X_train_bs, y_train_bs, n_more_iter=step_size)
        y_pred = fm.predict(X_test)
        rmse_train.append(np.sqrt(mean_squared_error(fm.predict(X_train_bs), y_train_bs)))
        rmse_test.append(np.sqrt(mean_squared_error(fm.predict(X_test), y_test)))
        
        # weighted
        fm_weighted.fit(X_train, y_train, C, n_more_iter=step_size)
        y_pred = fm_weighted.predict(X_test)
        rmse_train_wtd.append(np.sqrt(mean_squared_error(fm_weighted.predict(X_train_bs), y_train_bs)))
        rmse_test_wtd.append(np.sqrt(mean_squared_error(fm_weighted.predict(X_test), y_test)))

    print ('------- restart ----------')
    rmse_test_re = []
    rmse_train_re = []
    
    
    for i in values:
        fm = als.FMRegression(n_iter=i, l2_reg_w=l2_reg_w,
                l2_reg_V=l2_reg_V, rank=rank, random_state=seed)
        fm.fit(X_train_bs, y_train_bs)
        rmse_test_re.append(np.sqrt(mean_squared_error(fm.predict(X_test), y_test)))
        rmse_train_re.append(np.sqrt(mean_squared_error(fm.predict(X_train_bs), y_train_bs)))

    from matplotlib import pyplot as plt

    with plt.style.context('fivethirtyeight'):
        plt.plot(values, rmse_train, label='train')
        plt.plot(values, rmse_test, label='test')
        plt.plot(values, rmse_train_re, label='train re', linestyle='--')
        plt.plot(values, rmse_test_re, label='test re', ls='--')
    plt.legend()
    plt.show()
    
    
    
    # weighted
    rmse_test_re_wtd = []
    rmse_train_re_wtd = []
    for i in values:
        fm_weighted = als_weighted.FMRegression_weighted(n_iter=i, l2_reg_w=l2_reg_w,
                l2_reg_V=l2_reg_V, rank=rank, random_state=seed)
        fm_weighted.fit(X_train, y_train, C)
        rmse_test_re_wtd.append(np.sqrt(mean_squared_error(fm_weighted.predict(X_test), y_test)))
        rmse_train_re_wtd.append(np.sqrt(mean_squared_error(fm_weighted.predict(X_train_bs), y_train_bs)))

    from matplotlib import pyplot as plt

    with plt.style.context('fivethirtyeight'):
        plt.plot(values, rmse_train_wtd, label='train')
        plt.plot(values, rmse_test_wtd, label='test')
        plt.plot(values, rmse_train_re_wtd, label='train re', linestyle='--')
        plt.plot(values, rmse_test_re_wtd, label='test re', ls='--')
    plt.legend()
    plt.show()
