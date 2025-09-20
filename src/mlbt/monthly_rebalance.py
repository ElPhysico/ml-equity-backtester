# src/mlbt/monthly_rebalance.py
import pandas as pd
from typing import Optional


def monthly_rebalance(px: pd.DataFrame, weights: Optional[dict] = None, cost_bps: float = 0.0) -> pd.Series:
    """
    Monthly rebalance strategy for the given tickers in px and given weights. If no weights are given, an equal weighted portfolio is assumed. Charges cost_bps (basis points) once at start and on each rebalance date.

    Returns equity series (index=date, values=equity).
    """
    s = pd.Series(px.index, index=px.index)
    rebal_dates = pd.DatetimeIndex(s.groupby(px.index.to_period('M')).last().values)
    
    first = px.dropna(axis=1, how="all").iloc[0:1]
    t0 = first.index[0]
    tickers = first.columns[first.notna().iloc[0]]
    assert len(tickers) > 0, "No tickers with data on t0"

    if weights:
        w = pd.Series(weights, dtype=float).reindex(tickers).fillna(0.0)
    else:
        w = pd.Series(1.0 / len(tickers), index=tickers)

    w = w.clip(lower=0).astype("float64")
    w /= w.sum()
    w_target = w.copy()

    # first day allocation cost
    eq = 1.0 - (cost_bps / 1e4) * w.abs().sum()
    equity = [eq]

    px = px.loc[t0:]
    daily_rets = px.ffill().pct_change().fillna(0.0)
    for date in px.index[1:]:
        r = daily_rets.loc[date, w.index].fillna(0.0)
        R_t = float((w * r).sum())
        eq *= (1.0 + R_t)
        w = (w * (1.0 + r)) / (1.0 + R_t)
        if date in rebal_dates:
            turnover = float((w - w_target).abs().sum())
            cost_frac = (cost_bps / 1e4) * turnover
            eq *= (1.0 - cost_frac)
            w = w_target
        
        equity.append(eq)
    
    eq = pd.Series(equity, index=px.index, name="equity")

    return eq