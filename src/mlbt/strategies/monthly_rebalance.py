# src/mlbt/strategies/monthly_rebalance.py
import pandas as pd
from mlbt.strategies.result import StrategyResult
from typing import Dict, Optional


def monthly_rebalance(px: pd.DataFrame,
                      weights: Optional[dict] = None,
                      cost_bps: float = 0.0,
                      name: Optional[str] = None
) -> StrategyResult:
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
    name : str, optional
        Name for the returned StrategyResult. If None, defaults to
        f"EW_RB_{ticker}" or f"W_RB_{ticker}" if weights are not None.

    Returns
    -------
    StrategyResult
        StrategyResult with daily equity, monthly rebal_dates, one-way turnover (excluding entry), and entry_cost_frac.

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
    turnover_items: Dict[pd.Timestamp, float] = {}

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

            turnover_items[date] = turnover
        # elif date == rebal_dates[-1]:
        #     # end of portfolio, we sell everything
            # eq *= (1.0 - cost_bps / 1e4)
        if not charged_init:
            # the very first transaction cost is included in first day return
            eq *= (1.0 - cost_bps / 1e4)
            charged_init = True
        
        equity.append(eq)

    if name is None:
        if len(tickers) < 3:
            ticker = "_".join(tickers.values)
        else:
            ticker = "multi_ticker"
        name = f"EW_BH_{ticker}"
    
    eq = pd.Series(equity, index=px.index, name=name)

    turnover_ser = (
        pd.Series(turnover_items, name="turnover")
        .sort_index()
        .astype("float64")
    )

    res = StrategyResult(
        name=name,
        equity=eq,
        rebal_dates=rebal_dates,
        turnover=turnover_ser,
        entry_cost_frac=cost_bps / 1e4,
        weights=weights if weights is not None else None,
        params={"cost_bps": cost_bps, "tickers": list(tickers)}
    )

    return res