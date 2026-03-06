
import torch
import torch.nn as nn

class LinearAR(nn.Module):
    def __init__(self, Dx=20, Dz=4):
        super().__init__()
        self.enc = nn.Linear(Dx, Dz, bias=False)
        self.dec = nn.Linear(Dz, Dx, bias=False)

    def forward(self, x_t):
        z_t = self.enc(x_t)
        xhat_next = self.dec(z_t)
        return xhat_next, z_t

class LinearAR2(nn.Module):
    """ AR(2): x_{t+1} <- [x_t, x_{t-1}] """
    def __init__(self, Dx=20, Dz=4):
        super().__init__()
        self.enc = nn.Linear(2 * Dx, Dz, bias=False)
        self.dec = nn.Linear(Dz, Dx, bias=False)

    def forward(self, x_t, x_tm1):
        x_cat = torch.cat([x_t, x_tm1], dim=-1)
        z = self.enc(x_cat)
        xhat_next = self.dec(z)
        return xhat_next, z

class LinearSeasonalAR(nn.Module):
    """ Seasonal AR with lag n: x_{t+n} <- x_t """
    def __init__(self, Dx=20, Dz=4):
        super().__init__()
        self.enc = nn.Linear(Dx, Dz, bias=False)
        self.dec = nn.Linear(Dz, Dx, bias=False)

    def forward(self, x_t):
        z = self.enc(x_t)
        xhat = self.dec(z)
        return xhat, z

class LinearARn(nn.Module):
    """ General AR(n): x_{t+1} <- [x_t, x_{t-1}, ..., x_{t-n+1}] """
    def __init__(self, Dx=20, Dz=4, n_lags=3):
        super().__init__()
        self.Dx = Dx
        self.n_lags = n_lags
        self.enc = nn.Linear(n_lags * Dx, Dz, bias=False)
        self.dec = nn.Linear(Dz, Dx, bias=False)

    def forward(self, x_lags):
        B = x_lags.shape[0]
        x_flat = x_lags.reshape(B, self.n_lags * self.Dx)
        z = self.enc(x_flat)
        xhat_next = self.dec(z)
        return xhat_next, z