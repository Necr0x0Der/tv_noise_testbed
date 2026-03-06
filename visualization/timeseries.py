
import numpy as np
import matplotlib.pyplot as plt


def plot_time_series_paper_style(sigma, env_snr, t, y_true, yhat_dict, dim=0, max_points=2000):
    """ Style roughly like the paper: long time axis, title shows Scale + env SNR. """
    lengths = [y_true.shape[0]] + [v.shape[0] for v in yhat_dict.values()]
    L = min(lengths)
    t = t[:L]
    y_true = y_true[:L]
    yhat_dict = {k: v[:L] for k, v in yhat_dict.items()}

    if len(t) > max_points:
        idx = np.linspace(0, len(t) - 1, max_points).astype(int)
        t = t[idx]
        y_true = y_true[idx]
        yhat_dict = {k: v[idx] for k, v in yhat_dict.items()}

    plt.figure(figsize=(12, 4.8))
    plt.plot(t, y_true[:, dim], linewidth=3, label="True")

    styles = {
        "AR": dict(linestyle="--", linewidth=2.0, alpha=0.65),
        "AR(2)": dict(linestyle=":", linewidth=2.2, alpha=0.8),
        "SAR": dict(linestyle="--", linewidth=2.8, alpha=0.9),
        "JEPA": dict(linestyle="-", linewidth=2.6, alpha=0.95),
        "VJEPA": dict(linestyle="--", linewidth=2.2, alpha=0.85),
        "BJEPA": dict(linestyle="-", linewidth=2.6, alpha=0.90),
        "VAE": dict(linestyle="-", linewidth=2.2, alpha=0.85),
        "PredVAE": dict(linestyle=":", linewidth=2.4, alpha=0.9),
    }

    for name, yhat in yhat_dict.items():
        kw = styles.get(name, dict(linestyle="-", linewidth=2.0, alpha=0.85))
        plt.plot(t, yhat[:, dim], label=name, **kw)

    plt.title(f"Scale {sigma:.1f} (SNR: {env_snr:.1f} dB)")
    plt.xlabel("Time Step (t)")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, ncol=3, loc="lower left")
    plt.tight_layout()
    plt.savefig(f"series_{sigma}.png")