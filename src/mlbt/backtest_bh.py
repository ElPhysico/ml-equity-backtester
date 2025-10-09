# src/mlbt/backtest_bh.py
"""
Buy-and-hold equity backtester.

This module provides a lightweight backtesting engine for strategies that buy and hold selected tickers over the full investment period. A one time, one-way allocation cost is applied to the returns from day 1 to day 2.
"""
import pandas as pd
from typing import Optional, Dict, Tuple
import math

from mlbt.strategies import StrategyResult


def backtest_bh(
    px_wide: pd.DataFrame,
    *,
    weights: Optional[Dict[str, float]] = None,
    cost_bps: float = 5.0,
    name: Optional[str] = None
) -> Tuple[StrategyResult, Dict]:
    first = px_wide.dropna(axis=1, how="all").iloc[0:1]
    t0 = first.index[0]
    tickers = first.columns[first.notna().iloc[0]]

    # some guards
    assert len(tickers) > 0, f"No tickers with data on t0 ({str(t0)})"
    if len(tickers) < len(px_wide.columns):
        raise ValueError(f"Tickers {set(px_wide.columns) - set(tickers)} are missing prices on t0 ({str(t0)})")
    
    # weights
    if weights is not None:
        total_weight = sum(weights.values())
        if not math.isclose(total_weight, 1):
            raise ValueError(f"sum of weights is not 1, is {total_weight}")
        w0 = pd.Series(weights).reindex(tickers, fill_value=0)
    else:
        w0 = pd.Series(1.0 / len(tickers), index=tickers, dtype="float64")

    # normalize
    norm = (px_wide / px_wide.loc[t0]).loc[t0:]
    norm = norm.reindex(columns=tickers).ffill()

    # calculate equity curve
    eq = (1.0 - cost_bps / 1e4) * (norm * w0).sum(axis=1)
    # reset eq on inital day to 1
    eq[t0] = 1.0
    
    if name is None:
        if len(tickers) < 3:
            ticker = "_".join(tickers.values)
        else:
            ticker = "multi_ticker"
        name = f"BH_{ticker}"
    eq.name = name

    # housekeeping for consistent meta data with other backtesting runs
    weights_map = {t0: w0}
    weights_df = (
        pd.DataFrame.from_dict(weights_map, orient="index")
        .sort_index()
        .fillna(0.0)
        .astype("float64")
    )
    weights_df.index.name = "rebal_date"
    weights_df.columns.name = "ticker"

    selections = {t0: tickers}

    turnover_items: Dict[pd.Timestamp, float] = {}
    turnover_ser = (
        pd.Series(turnover_items, name="turnover")
        .sort_index()
        .astype("float64")
    )

    params = {"cost_bps": cost_bps}

    rebal_dates = px_wide.index[[]]

    res = StrategyResult(
        name=name,
        equity=eq,
        rebal_dates=rebal_dates,
        turnover=turnover_ser,
        entry_cost_frac=cost_bps / 1e4,
        selections=selections,
        weights=weights_df,
        params=params
    )

    return res, params