
import math
import torch
import torch.nn.functional as F


def vicreg_loss(z1, z2, inv_coeff=25.0, var_coeff=25.0, cov_coeff=1.0, eps=1e-4):
    inv = F.mse_loss(z1, z2)

    def var_term(z):
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(F.relu(1.0 - std))

    var = var_term(z1) + var_term(z2)

    def cov_term(z):
        z = z - z.mean(dim=0, keepdim=True)
        N, D = z.shape
        cov = (z.T @ z) / (N - 1 + 1e-12)
        offdiag = cov - torch.diag(torch.diag(cov))
        return (offdiag ** 2).sum() / D

    cov = cov_term(z1) + cov_term(z2)
    return inv_coeff * inv + var_coeff * var + cov_coeff * cov


def diag_gaussian_kl(mu_q, logvar_q, mu_p=None, logvar_p=None):
    if mu_p is None:
        mu_p = torch.zeros_like(mu_q)
    if logvar_p is None:
        logvar_p = torch.zeros_like(logvar_q)
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-12) - 1.0)
    return kl.sum(dim=-1).mean()


def diag_gaussian_nll(sample_z, mu, logvar):
    var = torch.exp(logvar)
    nll = 0.5 * ((sample_z - mu) ** 2 / (var + 1e-12) + logvar + math.log(2 * math.pi))
    return nll.sum(dim=-1).mean()