# src/mlbt/notebooks/demonstration.py
"""
Small modules for demonstration purposes in notebooks.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlbt.simulator.simulators import simulate_universe
from mlbt.visualisation import plot_universe
from mlbt.calendar import tdy_from_index


def notebook_gbm_demo_universe() -> None:
    """
    Simulates and plots GBM paths for demonstration purposes.
    """
    sim_start = "2000-01-01"
    sim_end = "2009-12-31"
    calendar = pd.bdate_range(sim_start, sim_end, freq="B")
    tdy = tdy_from_index(calendar)

    from mlbt.specs.samples import univ_gbm_demo
    px, meta = simulate_universe(
        univ=univ_gbm_demo,
        calendar=calendar
    )
    univ_gbm_demo.meta = meta

    mu = univ_gbm_demo.sim_params["mu"][0]
    sigma = univ_gbm_demo.sim_params["sigma"][0]
    g = mu - 1/2 * sigma**2
    subtitle = f"Parameters: μ = {mu:.2%} | σ = {sigma:.2%} | g = {g:.2%}\n"
    subtitle += f"{univ_gbm_demo.n_tickers} tickers | TDY = {tdy}"

    _ = plot_universe(
        px=px,
        univ=univ_gbm_demo,
        title="Demo GBM paths",
        subtitle=subtitle
    )



def notebook_elasticNet_penalties(l1_ratio: float = 0.5) -> None:
    # Grid over coefficient space
    w1 = np.linspace(-2.0, 2.0, 400)
    w2 = np.linspace(-2.0, 2.0, 400)
    W1, W2 = np.meshgrid(w1, w2)

    # Example quadratic OLS loss (ellipses) centered at a non-axis point
    Z = (W1 - 1.2)**2 / 2 + (W2 - 0.8)**2
    levels = [0.2, 0.5, 1, 2, 3]

    # Constraint functions
    L2 = np.sqrt(W1**2 + W2**2)                            # Ridge: ||w||_2
    L1 = np.abs(W1) + np.abs(W2)                           # Lasso: ||w||_1
    EN = l1_ratio * L1 + (1 - l1_ratio) * 0.5 * (W1**2 + W2**2)

    # Plot trio
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    # Lasso panel
    axes[0].contour(W1, W2, Z, levels=levels, linestyles="dotted")
    axes[0].contour(W1, W2, L1, levels=[1], linewidths=2)
    axes[0].plot(1.2, 0.8, "o", ms=4)
    axes[0].set_title("Lasso (L¹) constraint")
    axes[0].set_aspect("equal")
    axes[0].axhline(0, lw=0.5); axes[1].axvline(0, lw=0.5)
    axes[0].set_xlabel(r"$w_1$"); axes[1].set_ylabel(r"$w_2$")

    # Ridge panel
    axes[1].contour(W1, W2, Z, levels=levels, linestyles="dotted")
    axes[1].contour(W1, W2, L2, levels=[1], linewidths=2)
    axes[1].plot(1.2, 0.8, "o", ms=4)
    axes[1].set_title("Ridge (L²) constraint")
    axes[1].set_aspect("equal")
    axes[1].axhline(0, lw=0.5); axes[0].axvline(0, lw=0.5)
    axes[1].set_xlabel(r"$w_1$"); axes[0].set_ylabel(r"$w_2$")

    # ElasticNet panel (rounded diamond)
    axes[2].contour(W1, W2, Z, levels=levels, linestyles="dotted")
    axes[2].contour(W1, W2, EN, levels=[1], linewidths=2)
    axes[2].plot(1.2, 0.8, "o", ms=4)
    axes[2].set_title(f"ElasticNet (L¹ + L²)  l1_ratio={l1_ratio}")
    axes[2].set_aspect("equal")
    axes[2].axhline(0, lw=0.5); axes[2].axvline(0, lw=0.5)
    axes[2].set_xlabel(r"$w_1$"); axes[2].set_ylabel(r"$w_2$")

    plt.suptitle("")
    plt.show()
