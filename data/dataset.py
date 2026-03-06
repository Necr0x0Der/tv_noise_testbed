# data/dataset.py
import torch

def make_lagged_tensor(x, n_lags):
    """
    x: [T, Dx]
    Returns: x_lags: [T-n_lags, n_lags, Dx], x_next: [T-n_lags, Dx]
    """
    T, Dx = x.shape
    xs = []
    for k in range(n_lags):
        xs.append(x[n_lags - 1 - k : T - 1 - k])
    x_lags = torch.stack(xs, dim=1)
    x_next = x[n_lags:]
    return x_lags, x_next

def make_train_test_split(x, s, train_T=6000, test_T=2000):
    assert x.shape[0] >= train_T + test_T
    x_tr = x[:train_T]
    s_tr = s[:train_T]
    x_te = x[train_T:train_T + test_T]
    s_te = s[train_T:train_T + test_T]
    return x_tr, s_tr, x_te, s_te