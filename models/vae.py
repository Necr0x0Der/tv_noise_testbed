# models/vae.py
import torch
import torch.nn as nn

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