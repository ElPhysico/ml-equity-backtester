# src/mlbt/notebooks/demonstration.py
"""
Small modules for demonstration purposes in notebooks.
"""
import pandas as pd

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