# src/mlbt/simulator/validation.py
"""
Validation tools for the simulator framework.
"""
import numpy as np
import pandas as pd

def _check_basic_sanity(px_wide: pd.DataFrame) -> None:
    # No NaNs / infs
    assert px_wide.to_numpy().dtype.kind == 'f' or px_wide.to_numpy().dtype.kind == 'i'
    assert np.isfinite(px_wide.to_numpy()).all(), "Found non-finite values in prices."
    # Positivity
    assert (px_wide.to_numpy() > 0).all(), "Prices must be strictly positive."
    # Monotonic time index
    assert isinstance(px_wide.index, pd.DatetimeIndex), "Index must be a DatetimeIndex."
    assert px_wide.index.is_monotonic_increasing, "Price index must be sorted ascending."

def _daily_log_returns(px_wide: pd.DataFrame) -> pd.DataFrame:
    return np.log(px_wide).diff().iloc[1:]  # drop first NaN row

def validate_gbm_scaling_and_drift(
    px_wide: pd.DataFrame,
    ann_mu: float,
    ann_sigma: float,
    trading_days_per_year: int = 252,
    *,
    mean_tol_std_errors: float = 4.0,
    std_tol_rel: float = 0.05,
) -> dict:
    """
    Validate that simulated daily log-returns match the target GBM drift/vol scaling.

    Parameters
    ----------
    px_wide : DataFrame
        Prices [dates x tickers], strictly positive.
    ann_mu : float
        Annualized drift used in the simulator.
    ann_sigma : float
        Annualized volatility used in the simulator.
    trading_days_per_year : int
        E.g., 252.
    mean_tol_std_errors : float
        Tolerance for mean drift in units of standard errors of the mean (SEM).
    std_tol_rel : float
        Relative tolerance for stdev, e.g. 0.05 allows ±5%.

    Returns
    -------
    report : dict
        Summary stats and boolean pass/fail flags.
    """
    _check_basic_sanity(px_wide)

    r = _daily_log_returns(px_wide)  # shape [T-1, N]
    T, N = r.shape

    # Targets under GBM
    mu_step = (ann_mu - 0.5 * ann_sigma**2) / trading_days_per_year
    sigma_step = float(ann_sigma / np.sqrt(trading_days_per_year))

    # Empiricals (pool across all tickers & days)
    r_vals = r.to_numpy().ravel()
    mean_emp = float(r_vals.mean())
    std_emp = float(r_vals.std(ddof=1))

    # Standard error of the mean (SEM)
    sem = float(std_emp / np.sqrt(r_vals.size))

    # Tolerances
    mean_within = abs(mean_emp - mu_step) <= mean_tol_std_errors * sem
    std_within = abs(std_emp - sigma_step) <= std_tol_rel * sigma_step

    # Per-ticker diagnostics (optional but useful)
    mean_per_ticker = r.mean(axis=0).to_numpy()
    std_per_ticker = r.std(axis=0, ddof=1).to_numpy()

    report = {
        "n_days": T,
        "n_tickers": N,
        "target": {"mu_step": mu_step, "sigma_step": sigma_step},
        "empirical": {"mean_all": mean_emp, "std_all": std_emp, "sem": sem},
        "per_ticker": {
            "mean_min_max": (float(mean_per_ticker.min()), float(mean_per_ticker.max())),
            "std_min_max": (float(std_per_ticker.min()), float(std_per_ticker.max())),
        },
        "pass": {
            "mean_drift": bool(mean_within),
            "std_scaling": bool(std_within),
        },
    }
    return report
