# models/vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearVAE(nn.Module):
    def __init__(self, Dx=20, Dz=4):
        super().__init__()
        self.mu = nn.Linear(Dx, Dz, bias=False)
        self.logvar = nn.Linear(Dx, Dz, bias=False)
        self.dec = nn.Linear(Dz, Dx, bias=False)

    def encode(self, x):
        mu = self.mu(x)
        logvar = self.logvar(x).clamp(-20, 20)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        xhat = self.dec(z)
        return xhat, mu, logvar

class LinearPredVAE(nn.Module):
    def __init__(self, Dx=20, Dz=4):
        super().__init__()
        self.mu = nn.Linear(Dx, Dz, bias=False)
        self.logvar = nn.Linear(Dx, Dz, bias=False)
        self.pred = nn.Linear(Dz, Dz, bias=False)
        self.dec = nn.Linear(Dz, Dx, bias=False)

    def encode(self, x):
        mu = self.mu(x)
        logvar = self.logvar(x).clamp(-20, 20)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_t):
        mu_t, logvar_t = self.encode(x_t)
        z_t = self.reparam(mu_t, logvar_t)
        xhat_t = self.dec(z_t)
        zhat_next = self.pred(mu_t)
        xhat_next = self.dec(zhat_next)
        return xhat_t, xhat_next, mu_t, logvar_t, z_t, zhat_next

class UnitRowLinear(nn.Module):
    '''
    y = W x, with constraint: each row w_i has ||w_i||_2 = 1 (hard, by parametrization).
    This prevents the encoder from shrinking components toward zero by scaling weights.
    '''
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False, eps: float = 1e-8):
        super().__init__()
        self.V = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.eps = eps

    def weight(self) -> torch.Tensor:
        V = self.V
        return V / (V.norm(dim=1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.weight()
        return F.linear(x, W, self.bias)

class LinearGatedPredAE(nn.Module):
    '''
    Dx=20 -> z8=8 projections -> reconstruct x_t
                      |
                      +-- select 4 dims (top-k) -> predict next 4D latents
    In a more general case, the full size of latents for reconstruction and
    the size of gated latent vector are hyperparameters
    (learning softer selection of signal components is to be developed)
    '''
    def __init__(self, Dx: int = 20, Dz8: int = 8, Dz4: int = 4, bias: bool = False):
        super().__init__()
        assert Dz8 == 8 and Dz4 == 4, "This implementation is set for the PCA-like 8->4 case."
        self.Dx, self.Dz8, self.Dz4 = Dx, Dz8, Dz4

        self.enc = UnitRowLinear(Dx, Dz8, bias=bias)  # projections
        self.dec = nn.Linear(Dz8, Dx, bias=bias)      # linear decoder (can be tied if desired)
        self.pred4 = nn.Linear(Dz4, Dz4, bias=bias)   # predictor on explicit 4D

        # gate selects which 4 of 8 are "signal"
        self.gate_logits = nn.Parameter(torch.zeros(Dz8))

    @torch.no_grad()
    def current_idx4(self) -> torch.Tensor:
        return torch.topk(self.gate_logits, self.Dz4).indices  # (4,)

    def topk_idx4(self) -> torch.Tensor:
        # hard selection (not differentiable w.r.t. logits; fine for EM-style updates)
        return torch.topk(self.gate_logits, self.Dz4).indices

    def forward(self, x_t: torch.Tensor, x_next: torch.Tensor):
        """
        x_t, x_next: (B, Dx) already whitened if you use Whitener outside.
        """
        mu8_t = self.enc(x_t)       # (B,8)
        mu8_n = self.enc(x_next)    # (B,8)

        # reconstruction uses all 8
        xhat_t = self.dec(mu8_t)

        # select 4 dims explicitly
        idx4 = self.topk_idx4()
        mu4_t = mu8_t.index_select(dim=1, index=idx4)  # (B,4)
        mu4_n = mu8_n.index_select(dim=1, index=idx4)  # (B,4)

        # predict next 4D latents
        mu4_n_hat = self.pred4(mu4_t)  # (B,4)

        mu8_next_hat = mu4_n_hat.new_zeros(mu4_n_hat.size(0), self.Dz8)  # (B,8)
        mu8_next_hat[:, idx4] = mu4_n_hat
        xhat_next = self.dec(mu8_next_hat)  # (B,Dx)

        return xhat_t, mu4_t, mu8_t, mu4_n, mu4_n_hat, xhat_next
