# Noisy-TV Toy Experiment: Linear Representation Learning

This repository provides a controlled, modular benchmarking environment for evaluating how different representation learning architectures handle the "Noisy-TV" problem. 

In this linear-Gaussian toy system, models must learn to encode a predictable, stable underlying signal while ignoring a highly stochastic distractor (the "Noisy TV") and observation noise. The benchmark evaluates the quality of the learned latent representations using a linear probe ($R^2$ score) against the true underlying signal states.



## 🗂️ Project Structure

The codebase is organized using a modular registry pattern to easily toggle experiments and scale to new architectures:

```text
noisy_tv_experiments/
│
├── main.py                  # Main entry point: runs the experiment sweep
├── config.py                # Global hyperparameters and configuration
│
├── utils/
│   ├── seed.py              # RNG seeding for reproducibility
│   ├── metrics.py           # R² score and Environment SNR calculation
│   └── regression.py        # OLS linear probe utilities
│
├── env/
│   └── noisy_tv.py          # Environment dynamics and dataset rollout
│
├── data/
│   └── dataset.py           # Time-series lagging and train/test splitting
│
├── models/
│   ├── vae.py               # Linear VAE and Predictive VAE
│   ├── ar.py                # Auto-Regressive models: AR(1), AR(2), Seasonal, AR(n)
│   ├── jepa.py              # Joint Embedding Predictive Architectures (VICReg, VJEPA)
│   └── bjepa.py             # Bayesian JEPA
│
├── training/
│   ├── trainer.py           # Generic full-batch training loop
│   ├── losses.py            # KL divergence, VICReg, and NLL loss functions
│   └── runners.py           # Isolated execution/training logic for each model
│
├── evaluation/
│   └── probes.py            # Linear probing execution
│
└── visualization/
    ├── timeseries.py        # Paper-style time-series plotting
    └── plots.py             # R² vs Sigma sweep plotting
```

## 🚀 Quickstart

### Prerequisites
Make sure you have a Python 3.8+ environment with the following dependencies installed:

```bash
pip install torch numpy pandas matplotlib
```

### Running Experiments
To run the full suite of models across a sweep of distractor noise scales (sigma), simply execute:

```bash
python main.py
```

This will output console logs of the R² scores for each sigma step and generate two sets of plots in your working directory:

1. `series_<sigma>.png`: Time-series plots mapping the true signal versus the predicted outputs for each active model at that specific noise scale.

2. `r2_vs_sigma.png`: A summary plot showing how well each model maintains its linear probe R² score as the distractor noise scale increases.

### 🛠️ Configuration & Ablations

Thanks to the Registry pattern, it is incredibly easy to turn models on or off without altering the core pipeline.

Open `main.py` and modify the `MODELS_TO_RUN` list to speed up development or isolate specific architectures:

```python
# Run all models:
MODELS_TO_RUN = None 

# Run a specific subset for faster debugging:
MODELS_TO_RUN = ["VAE", "JEPA", "BJEPA"]
```

You can tweak the environment dynamics (signal frequencies, transition matrices) in `env/noisy_tv.py`, 
and adjust default hyperparameters (learning rate, sequence length, hidden dimensions) in `config.py`.

### 🧩 Adding a New Model
To add a custom architecture to the benchmark, follow these three steps:

1. **Define the Model:** Create your PyTorch `nn.Module` in the `models/` directory (e.g., `models/my_model.py`).
2. **Create a Runner:** In `training/runners.py`, write a function `run_mymodel(x_tr, s_tr, x_te, s_te, device, steps, lr)` that handles data slicing, loss calculation, and the training loop. It must return the R² score and the predicted time-series tensor.
3. **Register It:** Add your new runner function to the `EXPERIMENT_REGISTRY` dictionary inside `main.py`. The pipeline will automatically handle the loop, metrics, and plotting for you.