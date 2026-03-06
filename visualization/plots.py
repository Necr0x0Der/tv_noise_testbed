
import pandas as pd
import matplotlib.pyplot as plt


def plot_r2_vs_sigma(df: pd.DataFrame, out_path: str = None):
    plt.figure()
    x = df["sigma"].to_numpy()

    for col, label in [
        ("R2_VAE", "VAE"),
        ("R2_AR", "AR(1)"),
        ("R2_AR2", "AR(2)"),
        ("R2_ARn", "AR(n)"),
        ("R2_SAR", "SAR"),
        ("R2_JEPA", "JEPA"),
        ("R2_VJEPA", "VJEPA"),
        ("R2_BJEPA", "BJEPA"),
        ("R2_PredVAE", "PredVAE"),
    ]:
        if col in df.columns:
            plt.plot(x, df[col].to_numpy(), marker="o", label=label)

    plt.xlabel("sigma")
    plt.ylabel("R² (linear probe)")
    plt.title("Noisy-TV Toy Experiment: R² vs sigma")
    plt.grid(True)
    plt.legend()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=200)