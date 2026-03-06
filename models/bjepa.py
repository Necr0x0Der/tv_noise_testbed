
import torch
import torch.nn as nn

class LinearBJEPA(nn.Module):
    def __init__(self, Dx=20, Dz=4):
        super().__init__()
        self.enc = nn.Linear(Dx, Dz, bias=False)
        self.mu_dyn = nn.Linear(Dz, Dz, bias=False)
        self.logvar_dyn = nn.Linear(Dz, Dz, bias=False)
        self.mu_prior = nn.Parameter(torch.zeros(Dz))
        self.logvar_prior = nn.Parameter(torch.zeros(Dz))
        self.mu_tgt = nn.Linear(Dx, Dz, bias=False)
        self.logvar_tgt = nn.Linear(Dx, Dz, bias=False)
        self._init_target()

    @torch.no_grad()
    def _init_target(self):
        self.mu_tgt.load_state_dict(self.enc.state_dict())
        nn.init.zeros_(self.logvar_tgt.weight)

    @torch.no_grad()
    def ema_update(self, tau=0.99):
        for p_tgt, p_enc in zip(self.mu_tgt.parameters(), self.enc.parameters()):
            p_tgt.data.mul_(tau).add_(p_enc.data, alpha=(1 - tau))

    def forward(self, x_t, x_next):
        z_t = self.enc(x_t)
        mu_dyn = self.mu_dyn(z_t)
        logvar_dyn = self.logvar_dyn(z_t).clamp(-20, 20)
        with torch.no_grad():
            mu_q = self.mu_tgt(x_next)
            logvar_q = self.logvar_tgt(x_next).clamp(-20, 20)
        return mu_dyn, logvar_dyn, mu_q, logvar_q

    def fused_posterior_mean(self, mu_dyn, logvar_dyn):
        var_dyn = torch.exp(logvar_dyn)
        var_prior = torch.exp(self.logvar_prior).unsqueeze(0)
        mu_prior = self.mu_prior.unsqueeze(0)
        return (var_prior * mu_dyn + var_dyn * mu_prior) / (var_dyn + var_prior + 1e-12)