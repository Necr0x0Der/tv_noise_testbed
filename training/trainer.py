
import torch

def train_full_batch(model, loss_fn, steps=6000, lr=1e-3, ema_tau=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model)
        loss.backward()
        opt.step()
        if ema_tau is not None and hasattr(model, "ema_update"):
            model.ema_update(tau=ema_tau)
    return model