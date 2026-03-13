
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

def plot_components_scatter(S, model_name, comp_x=0, comp_y=1, s=8, alpha=0.6, out_path: str = None):
    S_np = S[model_name]
    x = S_np[:, comp_x]
    y = S_np[:, comp_y]

    plt.figure(figsize=(6,6))
    plt.scatter(x, y, s=s, alpha=alpha)
    plt.xlabel(f"component {comp_x+1}")
    plt.ylabel(f"component {comp_y+1}")
    plt.title(f"{model_name} components {comp_x+1} and {comp_y+1}")
    plt.grid(alpha=0.2)
    plt.gca().set_aspect('equal', adjustable='box')
    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
    else:
        plt.show()
    plt.close()
