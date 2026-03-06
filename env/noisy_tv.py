# env/noisy_tv.py
import math
import torch


def column_unit_norm_(W: torch.Tensor, eps: float = 1e-12):
    col_norm = torch.sqrt((W * W).sum(dim=0, keepdim=True)).clamp_min(eps)
    return W / col_norm


def make_rotation_matrix(Ds, omega, device):
    """ Build a block-diagonal rotation matrix with frequency omega. Ds must be even. """
    assert Ds % 2 == 0
    R = torch.zeros(Ds, Ds, device=device)
    for i in range(0, Ds, 2):
        c = torch.cos(torch.tensor(omega, device=device))
        s = torch.sin(torch.tensor(omega, device=device))
        R[i:i + 2, i:i + 2] = torch.tensor([[c, -s], [s, c]], device=device)
    return R


@torch.no_grad()
def rollout_noisy_tv(
        Dx=20, Ds=4, Dd=4, T=8000, sigma=0.0, a_scale=0.99,
        w_std=0.3, v_std=0.3, eps_std=0.01, device="cpu", seed=111
):
    """
    s_{t+1} = (a_scale * Q) s_t + w_t
    d_{t+1} = 0.9 d_t + v_t
    x_t = C s_t + D (sigma d_t) + eps_t
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # orthogonal base rotation Q
    omega = 0.1 * math.pi
    Q = make_rotation_matrix(Ds, omega, device)
    A = a_scale * Q

    C = column_unit_norm_(torch.randn(Dx, Ds, generator=g, device=device))
    D = column_unit_norm_(torch.randn(Dx, Dd, generator=g, device=device))

    s = torch.zeros(T, Ds, device=device)
    d = torch.zeros(T, Dd, device=device)
    x = torch.zeros(T, Dx, device=device)

    x_sig = torch.zeros(T, Dx, device=device)
    x_noi = torch.zeros(T, Dx, device=device)

    s_prev = torch.randn(Ds, generator=g, device=device)
    d_prev = torch.randn(Dd, generator=g, device=device)

    for t in range(T):
        eps = eps_std * torch.randn(Dx, generator=g, device=device)
        sig_part = C @ s_prev
        noi_part = D @ (sigma * d_prev) + eps

        x[t] = sig_part + noi_part
        x_sig[t] = sig_part
        x_noi[t] = noi_part

        s[t] = s_prev
        d[t] = d_prev

        w = w_std * torch.randn(Ds, generator=g, device=device)
        v = v_std * torch.randn(Dd, generator=g, device=device)

        s_prev = A @ s_prev + w
        d_prev = 0.9 * d_prev + v

    return x, s, d, x_sig, x_noi, (A, C, D)