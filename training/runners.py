
import torch
import torch.nn.functional as F

from config import DX, DZ
from data.dataset import make_lagged_tensor
from models.vae import LinearVAE, LinearPredVAE
from models.ar import LinearAR, LinearAR2, LinearSeasonalAR, LinearARn
from models.jepa import LinearJEPA, LinearVJEPA
from models.bjepa import LinearBJEPA
from training.trainer import train_full_batch
from training.losses import diag_gaussian_kl, diag_gaussian_nll, vicreg_loss
from evaluation.probes import evaluate_linear_probe

def run_vae(x_tr, s_tr, x_te, s_te, device, steps, lr):
    x_t_tr, s_t_tr, s_n_tr = x_tr[:-1], s_tr[:-1], s_tr[1:]
    x_t_te, s_t_te, s_n_te = x_te[:-1], s_te[:-1], s_te[1:]

    vae = LinearVAE(Dx=DX, Dz=DZ).to(device)
    def vae_loss(m):
        xhat, mu, logvar = m(x_tr)
        return F.mse_loss(xhat, x_tr) + 1.0 * diag_gaussian_kl(mu, logvar)

    train_full_batch(vae, vae_loss, steps=steps, lr=lr)

    with torch.no_grad():
        mu_tr, _ = vae.encode(x_t_tr)
        mu_te, _ = vae.encode(x_t_te)

    r2, _ = evaluate_linear_probe(
        mu_tr.cpu().numpy(), s_t_tr.cpu().numpy(),
        mu_te.cpu().numpy(), s_t_te.cpu().numpy()
    )
    # Fit on next-step for plotting parity with predictive models
    _, y_pred_plot = evaluate_linear_probe(
        mu_tr.cpu().numpy(), s_n_tr.cpu().numpy(),
        mu_te.cpu().numpy(), s_n_te.cpu().numpy()
    )
    return r2, y_pred_plot

def run_predvae(x_tr, s_tr, x_te, s_te, device, steps, lr):
    x_t_tr, x_n_tr, s_n_tr = x_tr[:-1], x_tr[1:], s_tr[1:]
    x_t_te, s_n_te = x_te[:-1], s_te[1:]

    predvae = LinearPredVAE(Dx=DX, Dz=DZ).to(device)
    def predvae_loss(m):
        xhat_t, xhat_next, mu, logvar, _, _ = m(x_t_tr)
        return (F.mse_loss(xhat_t, x_t_tr) + 1.0 * F.mse_loss(xhat_next, x_n_tr) +
                1.0 * diag_gaussian_kl(mu, logvar))

    train_full_batch(predvae, predvae_loss, steps=steps, lr=lr)

    with torch.no_grad():
        _, _, _, _, _, zhat_p_tr = predvae(x_t_tr)
        _, _, _, _, _, zhat_p_te = predvae(x_t_te)

    return evaluate_linear_probe(
        zhat_p_tr.cpu().numpy(), s_n_tr.cpu().numpy(),
        zhat_p_te.cpu().numpy(), s_n_te.cpu().numpy()
    )

def run_rnd_proj(x_tr, s_tr, x_te, s_te, device, steps, lr):
    return run_vae(x_tr, s_tr, x_te, s_te, device, steps=0, lr=0)

def pca_comp(x_tr, s_tr, x_te, s_te, c1=0, c2=4):
    x_t_tr, x_n_tr, s_n_tr = x_tr[:-1], x_tr[1:], s_tr[1:]
    x_t_te, s_n_te = x_te[:-1], s_te[1:]
    _, _, V_k = torch.pca_lowrank(x_t_tr, q=8, center=True)
    V_k = V_k[:,c1:c2]
    def pca(x):
        return x @ V_k ### (- x.mean(0))
    return evaluate_linear_probe(
        pca(x_t_tr).cpu().numpy(), s_n_tr.cpu().numpy(),
        pca(x_t_te).cpu().numpy(), s_n_te.cpu().numpy()
    )

def run_pca4(x_tr, s_tr, x_te, s_te, device, steps, lr):
    return pca_comp(x_tr, s_tr, x_te, s_te, 0, 4)

def run_pca8(x_tr, s_tr, x_te, s_te, device, steps, lr):
    return pca_comp(x_tr, s_tr, x_te, s_te, 4, 8)


def run_ar(x_tr, s_tr, x_te, s_te, device, steps, lr):
    x_t_tr, x_n_tr, s_n_tr = x_tr[:-1], x_tr[1:], s_tr[1:]
    x_t_te, s_n_te = x_te[:-1], s_te[1:]

    ar = LinearAR(Dx=DX, Dz=DZ).to(device)
    def ar_loss(m):
        xhat_next, _ = m(x_t_tr)
        return F.mse_loss(xhat_next, x_n_tr)

    train_full_batch(ar, ar_loss, steps=steps, lr=lr)

    with torch.no_grad():
        _, ztr = ar(x_t_tr)
        _, zte = ar(x_t_te)

    return evaluate_linear_probe(
        ztr.cpu().numpy(), s_n_tr.cpu().numpy(),
        zte.cpu().numpy(), s_n_te.cpu().numpy()
    )

def run_ar2(x_tr, s_tr, x_te, s_te, device, steps, lr):
    x_tm1_tr, x_t_tr2, x_n_tr2, s_n_tr2 = x_tr[:-2], x_tr[1:-1], x_tr[2:], s_tr[2:]
    x_tm1_te, x_t_te2, s_n_te2 = x_te[:-2], x_te[1:-1], s_te[2:]

    ar2 = LinearAR2(Dx=DX, Dz=DZ).to(device)
    def ar2_loss(m):
        xhat_next, _ = m(x_t_tr2, x_tm1_tr)
        return F.mse_loss(xhat_next, x_n_tr2)

    train_full_batch(ar2, ar2_loss, steps=steps, lr=lr)

    with torch.no_grad():
        _, ztr_2 = ar2(x_t_tr2, x_tm1_tr)
        _, zte_2 = ar2(x_t_te2, x_tm1_te)

    return evaluate_linear_probe(
        ztr_2.cpu().numpy(), s_n_tr2.cpu().numpy(),
        zte_2.cpu().numpy(), s_n_te2.cpu().numpy()
    )

def run_sar(x_tr, s_tr, x_te, s_te, device, steps, lr):
    lag = 35
    x_t_tr_s, x_n_tr_s, s_n_tr_s = x_tr[:-lag], x_tr[lag:], s_tr[lag:]
    x_t_te_s, s_n_te_s = x_te[:-lag], s_te[lag:]

    sar = LinearSeasonalAR(Dx=DX, Dz=DZ).to(device)
    def sar_loss(m):
        xhat, _ = m(x_t_tr_s)
        return F.mse_loss(xhat, x_n_tr_s)

    train_full_batch(sar, sar_loss, steps=steps, lr=lr)

    with torch.no_grad():
        _, ztr_s = sar(x_t_tr_s)
        _, zte_s = sar(x_t_te_s)

    return evaluate_linear_probe(
        ztr_s.cpu().numpy(), s_n_tr_s.cpu().numpy(),
        zte_s.cpu().numpy(), s_n_te_s.cpu().numpy()
    )

def run_arn(x_tr, s_tr, x_te, s_te, device, steps, lr):
    n_lags = 4
    x_lags_tr, x_n_tr_n = make_lagged_tensor(x_tr, n_lags)
    x_lags_te, _ = make_lagged_tensor(x_te, n_lags)
    s_n_tr_n, s_n_te_n = s_tr[n_lags:], s_te[n_lags:]

    arn = LinearARn(Dx=DX, Dz=DZ, n_lags=n_lags).to(device)
    def arn_loss(m):
        xhat_next, _ = m(x_lags_tr)
        return F.mse_loss(xhat_next, x_n_tr_n)

    train_full_batch(arn, arn_loss, steps=steps, lr=lr)

    with torch.no_grad():
        _, ztr_n = arn(x_lags_tr)
        _, zte_n = arn(x_lags_te)

    return evaluate_linear_probe(
        ztr_n.cpu().numpy(), s_n_tr_n.cpu().numpy(),
        zte_n.cpu().numpy(), s_n_te_n.cpu().numpy()
    )

def run_jepa(x_tr, s_tr, x_te, s_te, device, steps, lr):
    x_t_tr, x_n_tr, s_n_tr = x_tr[:-1], x_tr[1:], s_tr[1:]
    x_t_te, x_n_te, s_n_te = x_te[:-1], x_te[1:], s_te[1:]

    jepa = LinearJEPA(Dx=DX, Dz=DZ).to(device)
    def jepa_loss(m):
        zhat_next, _, ztgt_next = m(x_t_tr, x_n_tr)
        return vicreg_loss(zhat_next, ztgt_next, 25.0, 25.0, 1.0)

    train_full_batch(jepa, jepa_loss, steps=steps, lr=lr, ema_tau=0.99)

    with torch.no_grad():
        zhat_tr, _, _ = jepa(x_t_tr, x_n_tr)
        zhat_te, _, _ = jepa(x_t_te, x_n_te)

    return evaluate_linear_probe(
        zhat_tr.cpu().numpy(), s_n_tr.cpu().numpy(),
        zhat_te.cpu().numpy(), s_n_te.cpu().numpy()
    )

def run_vjepa(x_tr, s_tr, x_te, s_te, device, steps, lr):
    x_t_tr, x_n_tr, s_n_tr = x_tr[:-1], x_tr[1:], s_tr[1:]
    x_t_te, x_n_te, s_n_te = x_te[:-1], x_te[1:], s_te[1:]

    vjepa = LinearVJEPA(Dx=DX, Dz=DZ).to(device)
    def vjepa_loss(m):
        mu_p, logvar_p, mu_q, logvar_q = m(x_t_tr, x_n_tr)
        std_q = torch.exp(0.5 * logvar_q)
        z_samp = mu_q + torch.randn_like(std_q) * std_q
        nll = diag_gaussian_nll(z_samp, mu_p, logvar_p)
        kl = diag_gaussian_kl(mu_q, logvar_q)
        return nll + 0.01 * kl

    train_full_batch(vjepa, vjepa_loss, steps=steps, lr=lr, ema_tau=0.99)

    with torch.no_grad():
        mu_p_tr, _, _, _ = vjepa(x_t_tr, x_n_tr)
        mu_p_te, _, _, _ = vjepa(x_t_te, x_n_te)

    return evaluate_linear_probe(
        mu_p_tr.cpu().numpy(), s_n_tr.cpu().numpy(),
        mu_p_te.cpu().numpy(), s_n_te.cpu().numpy()
    )

def run_bjepa(x_tr, s_tr, x_te, s_te, device, steps, lr):
    x_t_tr, x_n_tr, s_n_tr = x_tr[:-1], x_tr[1:], s_tr[1:]
    x_t_te, x_n_te, s_n_te = x_te[:-1], x_te[1:], s_te[1:]

    bjepa = LinearBJEPA(Dx=DX, Dz=DZ).to(device)
    def bjepa_loss(m):
        mu_dyn, logvar_dyn, mu_q, logvar_q = m(x_t_tr, x_n_tr)
        std_q = torch.exp(0.5 * logvar_q)
        z_samp = mu_q + torch.randn_like(std_q) * std_q
        nll = diag_gaussian_nll(z_samp, mu_dyn, logvar_dyn)
        kl_target = diag_gaussian_kl(mu_q, logvar_q)
        kl_prior = diag_gaussian_kl(m.mu_prior.unsqueeze(0), m.logvar_prior.unsqueeze(0))
        return nll + 0.01 * kl_target + 0.1 * kl_prior

    train_full_batch(bjepa, bjepa_loss, steps=steps, lr=lr, ema_tau=0.99)

    with torch.no_grad():
        mu_dyn_tr, logvar_dyn_tr, _, _ = bjepa(x_t_tr, x_n_tr)
        mu_dyn_te, logvar_dyn_te, _, _ = bjepa(x_t_te, x_n_te)
        mu_post_tr = bjepa.fused_posterior_mean(mu_dyn_tr, logvar_dyn_tr)
        mu_post_te = bjepa.fused_posterior_mean(mu_dyn_te, logvar_dyn_te)

    return evaluate_linear_probe(
        mu_post_tr.cpu().numpy(), s_n_tr.cpu().numpy(),
        mu_post_te.cpu().numpy(), s_n_te.cpu().numpy()
    )
