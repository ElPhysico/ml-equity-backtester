# src/mlbt/backtest_mr.py
"""
Monthly rebalance backtester.

This module provides a convenience wrapper for the Top-N selection backtester to backtest a monthly rebalance strategy on the whole provided universe.
"""
import pandas as pd

from .backtest_topn import backtest_topn
from mlbt.strategy_result import StrategyResult
from mlbt.calendar import build_month_end_grid


def backtest_mr(
    px_wide: pd.DataFrame,
    *,
    cost_bps: float = 5.0,
    strict: bool = True,
    name: str | None = None
) -> tuple[StrategyResult, dict]:
    """
    Convenience wrapper for a monthly rebalance strategy using `backtest_topn`. Currently only supports equal weight across entire provided universe.

    Parameters
    ----------
    px_wide : pandas.DataFrame
        Wide DataFrame of daily prices, with a DatetimeIndex and one column per
        ticker.  The index is used to infer month-end rebalance dates.
    
    cost_bps : float, default=5.0
        One-way transaction cost in basis points applied on turnover at each
        rebalance.  A cost of 5.0 corresponds to 0.05 %.

    strict: bool, optional True
        Currently not used, may be used to enforce behavior in case of missing tickers in the future (hold, cash, ...).

    name : str, optional
        Optional identifier for the resulting strategy.  Defaults to
        `"MR_EW"`.

    Returns
    -------
    (res, backtest_params) : tuple[StrategyResult, dict]
        `res`: StrategyResult of the MR backtest.
        `backtest_params`: Dictionary containing the backtest params used.
    """
    # create arbitraty predictions table for compatibility
    me_grid = build_month_end_grid(px_wide)
    me_grid["y_pred"] = 1

    res, meta = backtest_topn(
        px_wide=px_wide,
        predictions=me_grid,
        rank_col="y_pred",
        N=len(px_wide.columns), # include all tickers
        cost_bps=cost_bps,
        strict=strict,
        name=name
    )

    return res, meta