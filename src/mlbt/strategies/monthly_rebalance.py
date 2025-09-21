# src/mlbt/strategies/monthly_rebalance.py
import pandas as pd
from typing import Optional


def monthly_rebalance(px: pd.DataFrame, weights: Optional[dict] = None, cost_bps: float = 0.0) -> pd.Series:
    """
    Equal-weight (or user-specified) monthly rebalance:
    - Identify month-end trading days and rebalance on each of those dates.
    - If `weights` is None, target is equal-weight across tickers available at t0.
      Otherwise, `weights` is a dict of target weights; missing tickers get 0.
    - Apply one-way cost_bps on turnover at each rebalance (and once at initial entry).

    Parameters
    ----------
    px : pd.DataFrame
        Daily close prices. Index: DatetimeIndex (tz-naive), ascending unique.
        Columns: ticker symbols. May contain NaNs; only tickers priced at t0 are eligible.
    weights : dict, optional
        Target weights by ticker. If None, uses equal-weight across eligible tickers.
        Negative weights are clipped to 0; weights are normalized to sum to 1.
    cost_bps : float, default 0.0
        One-way transaction cost in basis points. Charged once at start and on
        each rebalance date as turnover * (cost_bps/1e4).

    Returns
    -------
    pd.Series
        Daily equity curve (post-cost), index = trading dates from t0 onward,
        name = 'equity'.

    Notes
    -----
    - Rebalance dates are the last trading day of each calendar month derived via
      `to_period('M').last()`.
    - Turnover is computed as the L1 distance between current and target weights:
      sum(|w_current - w_target|). The cost applied is turnover * (cost_bps/1e4).
    - Initial entry cost is embedded in the first day's return (then eq[t0] is 1.0 implicitly via path).
    - Between rebalances, weights drift with relative returns and are renormalized
      using standard self-financing accounting.
    - Asserts at least one ticker has a finite price at t0.
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

    eq = 1.0
    equity = [eq]
    charged_init = False

    px = px.loc[t0:]
    daily_rets = px.ffill().pct_change().fillna(0.0)
    for date in px.index[1:]:
        r = daily_rets.loc[date, w.index].fillna(0.0)
        R_t = float((w * r).sum())
        eq *= (1.0 + R_t)
        w = (w * (1.0 + r)) / (1.0 + R_t)
        if date in rebal_dates[:-1]: # no rebalance on last rebalance date
            # rebalance
            turnover = float((w - w_target).abs().sum())
            cost_frac = (cost_bps / 1e4) * turnover
            eq *= (1.0 - cost_frac)
            w = w_target
        # elif date == rebal_dates[-1]:
        #     # end of portfolio, we sell everything
            # eq *= (1.0 - cost_bps / 1e4)
        if not charged_init:
            # the very first transaction cost is included in first day return
            eq *= (1.0 - cost_bps / 1e4)
            charged_init = True
        
        equity.append(eq)
    
    eq = pd.Series(equity, index=px.index, name="equity")

    return eq