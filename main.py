
import torch
import numpy as np
import pandas as pd

# from config import DX, DS, DD, DEFAULT_SEED, DEFAULT_LR, DEFAULT_TRAIN_STEPS
from config import *
from utils.seed import set_seed
from utils.metrics import env_snr_db
from env.noisy_tv import rollout_noisy_tv
from data.dataset import make_train_test_split
from visualization.timeseries import plot_time_series_paper_style
from visualization.plots import plot_r2_vs_sigma

# Import our isolated runners
from training.runners import (
    run_vae, run_predvae, run_ar, run_ar2, run_sar,
    run_rnd_proj, run_pca4, run_pca8,
    run_arn, run_jepa, run_vjepa, run_bjepa
)

# A registry mapping names to their execution functions
EXPERIMENT_REGISTRY = {
    "VAE": run_vae,
    "RNDproj": run_rnd_proj,
    "PredVAE": run_predvae,
    "PCA(0-3)": run_pca4,
    "PCA(4-7)": run_pca8,
    "AR(1)": run_ar,
    "AR(2)": run_ar2,
    "SAR": run_sar,
    "AR(n)": run_arn,
    "JEPA": run_jepa,
    "VJEPA": run_vjepa,
    "BJEPA": run_bjepa,
}


def run_sigma(
        sigma, active_models=None, device="cpu", seed=DEFAULT_SEED,
        train_T=6000, test_T=2000, a_scale=0.98, w_std=0.3,
        steps=DEFAULT_TRAIN_STEPS, lr=DEFAULT_LR,
        make_timeseries=False, ts_points=2000, ts_dim=0,
):
    # Default to running all models if none specified
    if active_models is None:
        active_models = list(EXPERIMENT_REGISTRY.keys())

    set_seed(seed)

    # 1. Environment Rollout & Prep
    x, s, d, x_sig, x_noi, _ = rollout_noisy_tv(
        Dx=DX, Ds=DS, Dd=DD, T=train_T + test_T, sigma=sigma,
        a_scale=a_scale, w_std=w_std, device=device, seed=seed
    )
    env_snr = env_snr_db(x_sig, x_noi)
    x_tr, s_tr, x_te, s_te = make_train_test_split(x, s, train_T=train_T, test_T=test_T)

    results = {"sigma": float(sigma), "env_snr_db": float(env_snr)}
    preds = {}

    # 2. Loop through only the activated models
    for model_name in active_models:
        if model_name in EXPERIMENT_REGISTRY:
            runner_fn = EXPERIMENT_REGISTRY[model_name]
            r2, y_pred = runner_fn(x_tr, s_tr, x_te, s_te, device, steps, lr)
            results[model_name] = float(r2)
            preds[model_name] = y_pred

    # 3. Visualization
    if make_timeseries:
        s_n_te = s_te[1:]  # Base temporal alignment for the true plot target
        t_np = np.arange(s_n_te.shape[0])
        plot_time_series_paper_style(
            sigma=sigma, env_snr=env_snr, t=t_np, y_true=s_n_te.cpu().numpy(),
            yhat_dict=preds, dim=ts_dim, max_points=ts_points
        )

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sigmas = np.linspace(0.0, 10.0, 11)

    # Easily toggle models to run ablations
    MODELS_TO_RUN = ["VAE", "PCA(0-3)", "PCA(4-7)", "JEPA", "BJEPA"]  # Change to None to run all

    rows = []
    for s in sigmas:
        row = run_sigma(
            sigma=float(s), active_models=MODELS_TO_RUN, device=device,
            seed=DEFAULT_SEED, train_T=5000, test_T=300, a_scale=0.99,
            w_std=0.1, steps=10000, lr=1e-3, make_timeseries=True
        )
        print(row)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("sigma").reset_index(drop=True)
    plot_r2_vs_sigma(df, MODELS_TO_RUN, out_path="r2_vs_sigma.png")

    print("\nSummary:\n", df)


if __name__ == "__main__":
    main()
