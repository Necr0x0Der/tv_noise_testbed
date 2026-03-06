# utils/metrics.py
import numpy as np
import torch

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))

def env_snr_db(x_signal: torch.Tensor, x_noise: torch.Tensor, eps: float = 1e-12) -> float:
    """ Environment SNR in observation space (dB): SNR = Var(signal part of x) / Var(noise part of x) """
    sig = x_signal.detach().cpu().numpy().reshape(-1)
    noi = x_noise.detach().cpu().numpy().reshape(-1)
    return float(10.0 * np.log10((np.var(sig) + eps) / (np.var(noi) + eps)))