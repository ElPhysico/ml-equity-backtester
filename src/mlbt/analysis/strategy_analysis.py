# src/mlbt/analysis/strategy_analysis.py
"""
This module holds analysis methods for strategy run data.
"""
import pandas as pd
import numpy as np
import scipy.stats as st
from statsmodels.stats.proportion import proportion_confint
from collections.abc import Sequence

from mlbt.specs.metrics import MetricsResult


def average_metricsresults(
    v: Sequence[MetricsResult],
    bootstrap_vol: bool = False
) -> pd.Series:
    df = pd.DataFrame([m.to_dict() for m in v]).drop(columns=["freq"], errors="ignore")
    s = df.mean(numeric_only=True)

    if bootstrap_vol:
        ci = st.bootstrap((df["vol_ann"].to_numpy(),), np.mean,
                        method="percentile", random_state=0).confidence_interval
        s["vol_ann_low"] = float(ci.low)
        s["vol_ann_high"] = float(ci.high)
    return s


def statistics_ann_log_growth(x: pd.Series, alt: str = "greater", cl: float = 0.95):
    arr = x.to_numpy()
    n = arr.size
    # mean + bootstrap CI
    b = st.bootstrap((arr,), np.mean, confidence_level=cl, n_resamples=10_000, method="percentile")

    # win-rate
    k = int((arr > 0).sum())
    p_hat = k / n if n else np.nan

    # Wilson CI
    p_lo, p_hi = proportion_confint(k, n, alpha=1 - cl, method="wilson") if n else (np.nan, np.nan)
    
    # Binomial (sign) test against 0.5 in the chosen direction
    p_val = st.binomtest(k, n, p=0.5, alternative=alt).pvalue if n else np.nan

    return pd.Series({
        "d": arr.mean(),
        "d_lo": b.confidence_interval.low,
        "d_hi": b.confidence_interval.high,
        "win_rate": p_hat,
        "win_lo": p_lo,
        "win_hi": p_hi,
        "p_value": p_val,
        "n": n,
        "k": k,
    })