# src/mlbt/simulator/trajectories.py
"""
Module to create different kinds of equity trajectories.
"""
import numpy as np
import pandas as pd
import string

from typing import Any

from mlbt.simulator.meta import build_trajectories_meta


def generate_unique_tickers(
    rng: np.random.Generator,
    lengths: list[int]
) -> list[str]:
    """
    Generate unique random ticker symbols for the given list of lengths.
    """
    tickers = set()
    alphabet = np.array(list(string.ascii_uppercase))
    for L in lengths:
        while True:
            chars = rng.choice(alphabet, size=L)
            ticker = ''.join(chars)
            if ticker not in tickers:
                tickers.add(ticker)
                break
    return list(tickers)


def simulate_gbm_trajectories(
    n_tickers: int,
    calendar: pd.DatetimeIndex,
    *,
    ann_mu: float = 0.05,
    ann_sigma: float = 0.1,
    trading_days_per_year: int = 252,
    seed: int = 0
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)

    dt = 1/trading_days_per_year

    lnS_0 = np.log(rng.integers(low=1, high=1000, size=n_tickers, endpoint=True))[:, None]
    mu = ann_mu * dt
    sigma = ann_sigma * np.sqrt(dt)   

    n = len(calendar)
    t = np.arange(1, n)
    Z = rng.standard_normal((n_tickers, n - 1))
    W = np.cumsum(Z, axis=1)
    lnS = lnS_0 + (mu - 1/2 * sigma**2) * t[None, :] + sigma * W

    S = np.exp(np.concatenate([lnS_0, lnS], axis=1))

    tickers = generate_unique_tickers(rng, rng.integers(low=2, high=5, size=n_tickers))
    px_wide = pd.DataFrame(S.T, index=calendar, columns=tickers)

    meta = build_trajectories_meta(
        n_tickers=n_tickers,
        calendar=calendar,
        params={
            "ann_mu": ann_mu,
            "ann_sigma": ann_sigma,
            "TDY": trading_days_per_year
        },
        seed=seed
    )
    
    return px_wide, meta
