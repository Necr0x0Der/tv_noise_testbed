# utils/regression.py
import numpy as np

def ols_fit_predict(X_train, Y_train, X_test):
    Xtr = np.asarray(X_train)
    Ytr = np.asarray(Y_train)
    Xte = np.asarray(X_test)
    Xtr_aug = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], axis=1)
    Xte_aug = np.concatenate([Xte, np.ones((Xte.shape[0], 1))], axis=1)
    W = np.linalg.lstsq(Xtr_aug, Ytr, rcond=None)[0]
    return Xte_aug @ W