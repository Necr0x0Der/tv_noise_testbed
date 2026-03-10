
import pandas as pd
import matplotlib.pyplot as plt


def plot_r2_vs_sigma(df: pd.DataFrame, models, out_path: str = None):
    plt.figure()
    x = df["sigma"].to_numpy()

    for col in df.columns:
        if col in models:
            plt.plot(x, df[col].to_numpy(), marker="o", label=col)

    plt.xlabel("sigma")
    plt.ylabel("R² (linear probe)")
    plt.title("Noisy-TV Toy Experiment: R² vs sigma")
    plt.grid(True)
    plt.legend()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=200)