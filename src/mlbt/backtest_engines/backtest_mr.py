# src/mlbt/backtest_mr.py
"""
Monthly rebalance backtester.


"""
import pandas as pd
import numpy as np
import math
import logging

from mlbt.specs.strategy_result import StrategyResult


def backtest_mr(
    px_wide: pd.DataFrame,
    *,
    weights: dict[str, float] | None = None,
    cost_bps: float = 5.0,
    strict: bool = True,
    name: str | None = None
) -> tuple[StrategyResult, dict]:
    """
    Backtest a monthly weighted rebalance strategy.

    Parameters
    ----------
    px_wide : pandas.DataFrame
        Wide DataFrame of daily prices, with a DatetimeIndex and one column per
        ticker.  The index is used to infer month-end rebalance dates.

    weights: dict[str, float], optional
        Dictionary containing weights for tickers in px. Missing tickers will get weight=0. If None, equal weights are assumed among all tickers.

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
    first = px_wide.dropna(axis=1, how="all").iloc[0:1]
    t0 = first.index[0]
    tickers = first.columns[first.notna().iloc[0]]

    # some guards
    assert len(tickers) > 0, f"No tickers with data on t0 ({str(t0)})"
    if len(tickers) < len(px_wide.columns):
        raise ValueError(f"Tickers {set(px_wide.columns) - set(tickers)} are missing prices on t0 ({str(t0)})")
    
    # get rebalance dates
    s = pd.Series(px_wide.index, index=px_wide.index)
    rebal_dates = pd.DatetimeIndex(s.groupby(px_wide.index.to_period('M')).last().values)
    t0 = rebal_dates[0]
    # first is allocation day, last is strategy end day, remove at the end
    rebal_dates = rebal_dates[1:]

    # weights
    if weights is not None:
        total_weight = sum(weights.values())
        if not math.isclose(total_weight, 1):
            raise ValueError(f"sum of weights is not 1, is {total_weight}")
        w0 = pd.Series(weights).reindex(tickers, fill_value=0)
    else:
        w0 = pd.Series(1.0 / len(tickers), index=tickers, dtype="float64")

    # setting containers
    if name is None: # highly recommend passing a name for identification
        name = f"MR"
    equity = pd.Series(np.nan, index=px_wide.loc[t0:].index, name=name)
    turnover_items: dict[pd.Timestamp, float] = {}

    nz_idx = w0.index[w0.abs().gt(1e-12)]
    selections = {t0: nz_idx.tolist()}

    daily_rets = px_wide.pct_change().fillna(0.0)
    # check for unusual large number of nans -> not intensive
    row_month = daily_rets.index.to_period("M")
    nan_share_df = daily_rets.isna().groupby(row_month).mean()
    bad_mask = nan_share_df > 0.3
    if bad_mask.to_numpy().any():
        bad_preview = nan_share_df.where(bad_mask).stack().sort_values(ascending=False).head(10)
        logging.warning(
            f"High NaN-share detected: {bad_mask.to_numpy().sum()} (month,ticker) pairs "
            f"with >30% missing daily returns.\nPreview:\n{bad_preview}"
        )

    t_i = t0
    equity.loc[t0] = 1.0 - cost_bps / 1e4
    L = np.log1p(daily_rets).cumsum()
    w = w0.copy()

    for t in rebal_dates:
        # equity evolution for block
        t_ip1 = px_wide[px_wide.index > t_i].index[0] # finding trading day after first day of period bc first day of period is the same as last day of last period
        G = np.exp(L.loc[t_i:t, w.index] - L.loc[t_i, w.index])
        equity.loc[t_ip1:t] = equity[t_i] * (G * w).sum(axis=1).loc[t_ip1:]

        # weights evolution and turnover
        if t != rebal_dates[-1]: # catch strategy end day
            w_new = w0.copy()
            union = w.index.union(w_new.index)
            w = G.loc[t] * w
            w = w / w.sum()
            w_old_aligned = w.reindex(union).fillna(0.0).astype("float64")
            w_new_aligned = w_new.reindex(union).fillna(0.0).astype("float64")
            turnover = (w_new_aligned - w_old_aligned).abs().sum()
            cost_frac = (cost_bps / 1e4) * turnover
            equity.loc[t] *= (1.0 - cost_frac)

            w = w_new
            t_i = t

            # tracking
            turnover_items[t] = turnover
    equity[t0] = 1

    # accumulating data
    weights_map = {t0: w0}
    weights_df = (
        pd.DataFrame.from_dict(weights_map, orient="index")
        .sort_index()
        .fillna(0.0)
        .astype("float64")
    )
    weights_df.index.name = "rebal_date"
    weights_df.columns.name = "ticker"

    turnover_ser = (
        pd.Series(turnover_items, name="turnover")
        .sort_index()
        .astype("float64")
    )

    backtest_params = {
        "cost_bps": cost_bps,
        "strict": strict
    }

    res = StrategyResult(
        name=name,
        equity=equity,
        rebal_dates=rebal_dates[:-1],
        turnover=turnover_ser,
        entry_cost_frac=cost_bps / 1e4,
        selections=selections,
        weights=weights_df,
        params=backtest_params
    )
    
    return res, backtest_params