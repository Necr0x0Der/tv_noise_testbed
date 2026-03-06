
import torch
import torch.nn as nn

class LinearJEPA(nn.Module):
    def __init__(self, Dx=20, Dz=4):
        super().__init__()
        self.enc = nn.Linear(Dx, Dz, bias=False)
        self.pred = nn.Linear(Dz, Dz, bias=False)
        self.tgt = nn.Linear(Dx, Dz, bias=False)
        self._init_target()

    @torch.no_grad()
    def _init_target(self):
        self.tgt.load_state_dict(self.enc.state_dict())

    @torch.no_grad()
    def ema_update(self, tau=0.99):
        for p_tgt, p_enc in zip(self.tgt.parameters(), self.enc.parameters()):
            p_tgt.data.mul_(tau).add_(p_enc.data, alpha=(1 - tau))

    def forward(self, x_t, x_next):
        z_t = self.enc(x_t)
        zhat_next = self.pred(z_t)
        with torch.no_grad():
            ztgt_next = self.tgt(x_next)
        return zhat_next, z_t, ztgt_next

class LinearVJEPA(nn.Module):
    def __init__(self, Dx=20, Dz=4):
        super().__init__()
        self.enc = nn.Linear(Dx, Dz, bias=False)
        self.mu_pred = nn.Linear(Dz, Dz, bias=False)
        self.logvar_pred = nn.Linear(Dz, Dz, bias=False)
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
        mu_p = self.mu_pred(z_t)
        logvar_p = self.logvar_pred(z_t).clamp(-20, 20)
        with torch.no_grad():
            mu_q = self.mu_tgt(x_next)
            logvar_q = self.logvar_tgt(x_next).clamp(-20, 20)
        return mu_p, logvar_p, mu_q, logvar_q