# src/mlbt/backtest_engines/backtest_topn.py
"""
Top-N equity selection backtester.

This module provides a lightweight backtesting engine for strategies that
select the top-N equities each month based on an external ranking signal or
model prediction.  Rebalance dates are inferred from `px_wide` as the last
trading day of each month.  Between rebalances, portfolio weights drift with
daily returns and no transactions are made.

Key features
------------
- Uses the provided `rank_col` in `predictions` to select the N highest-ranked
  tickers per rebalance month.
- Applies per-rebalance one-way transaction costs (`cost_bps`) proportional to
  turnover.
- Checks for potential stock splits in price data via
  `mlbt.load_prices.flag_suspect_splits`.
- Detects months or tickers with unusually high shares of missing daily returns
  (e.g., holidays or delistings) and logs warnings.
- Returns a unified `StrategyResult` that can be evaluated with common
  performance metrics and plotted alongside benchmark strategies.

All input data should be pre-curated to exclude dividend effects or splits, or
should use adjusted prices to ensure return consistency.
"""
import pandas as pd
import numpy as np
import logging

from mlbt.specs.strategy_result import StrategyResult
from mlbt.utils import validate_columns_exist


def backtest_topn(
    px_wide: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    rank_col: str = "y_pred",
    N: int = 10,
    cost_bps: float = 5.0,
    strict: bool = True,
    name: str | None = None
) -> tuple[StrategyResult, dict]:
    """
    Backtest a monthly Top-N equity selection strategy.

    Each month the function selects the top-N tickers according to the column
    `rank_col` in the supplied `predictions` table, allocates equal weights,
    and holds them for one month.  Daily portfolio value is tracked using
    `px_wide` prices; weights drift between rebalances.  One-way transaction
    costs are applied on each rebalance proportional to turnover.

    Parameters
    ----------
    px_wide : pandas.DataFrame
        Wide DataFrame of daily prices, with a DatetimeIndex and one column per
        ticker.  The index is used to infer month-end rebalance dates.

    predictions : pandas.DataFrame
        MultiIndex DataFrame with index levels `['month', 'ticker']` containing
        at least the column specified by `rank_col`.  Each row represents the
        model signal or score for a given month and ticker.

    rank_col : str, default 'y_pred'
        Name of the column in `predictions` to rank tickers by.  Higher values
        correspond to stronger buy signals.

    N : int, default=10
        Number of tickers to select each month.

    cost_bps : float, default=5.0
        One-way transaction cost in basis points applied on turnover at each
        rebalance.  A cost of 5.0 corresponds to 0.05 %.

    strict: bool, optional True
        Currently not used, may be used to enforce behavior in case of missing tickers in the future (hold, cash, ...).

    name : str, optional
        Optional identifier for the resulting strategy.  Defaults to
        `"{rank_col}_top{N}"`.

    Returns
    -------
    (res, backtest_params) : tuple[StrategyResult, dict]
        `res`: StrategyResult of the Top-N backtest.
        `backtest_params`: Dictionary containing the backtest params used.

    Notes
    -----
    - The equity series starts at 1.0.  The initial transaction cost is applied
      on the **first trading day after** the initial rebalance, ensuring that
      the equity value on day one reflects pre-investment capital.
    - Raises a ValueError if fewer than `N` tickers are available in any month
      or if predictions are missing for required months.
    - Logs a warning if any ticker-month pair exhibits more than 30 % missing
      daily returns.
    """
    # some guards
    validate_columns_exist(predictions, rank_col)

    # aligning calendars -> not intensive
    m1 = predictions.index.get_level_values("month").sort_values()
    m1 = m1.append(pd.PeriodIndex([m1[-1] + 1])) # we predict for next month
    m2 = px_wide.index.to_period("M")
    common_months = m1.intersection(m2).unique()
    if len(common_months) == 0:
        raise ValueError("Calendars have no common months")
    first_month = common_months.min()
    last_month = common_months.max()
    px = px_wide[(px_wide.index.to_period("M") >= first_month) & (px_wide.index.to_period("M") <= last_month)]
    preds = predictions[(predictions.index.get_level_values("month") >= first_month) & (predictions.index.get_level_values("month") <= last_month)]

    # check if prediction for every rebalance month is available
    rebal_months = px.index.to_period("M").unique()
    pred_months = predictions.index.get_level_values("month").unique()
    missing = rebal_months[:-1].difference(pred_months)
    if not missing.empty:
        raise ValueError(f"{len(missing)} rebalance months do not have a prediction\n{missing[:5]}")

    # get rebalance dates
    s = pd.Series(px.index, index=px.index)
    rebal_dates = pd.DatetimeIndex(s.groupby(px.index.to_period('M')).last().values)
    t0 = rebal_dates[0]
    # first is allocation day, last is strategy end day, remove at the end
    rebal_dates = rebal_dates[1:]

    # topn map
    topn_tbl = preds.groupby("month", observed=True, group_keys=False).apply(lambda g: g.nlargest(N, columns=rank_col))
    
    # check for months with not enough tickers -> a bit intensive      
    counts = topn_tbl.groupby("month", observed=True).size()
    bad_months = counts[counts != N]
    if not bad_months.empty:
        raise ValueError(
            f"{len(bad_months)} months do not have exactly N={N} tickers, "
            f"examples:\n{bad_months.head(3)}"
        )
    
    topn_map: dict[pd.Period, pd.Index] = topn_tbl.reset_index().groupby("month", observed=True)["ticker"].apply(pd.Index).to_dict()
    
    # initial portfolio allocation on first rebalance date
    topn = topn_map[t0.to_period("M")]
    w = pd.Series(1.0 / len(topn), index=topn)
    
    # setting containers
    if name is None: # highly recommend passing a name for identification
        name = f"{rank_col}_top{N}"
    equity = pd.Series(np.nan, index=px.loc[t0:].index, name=name)
    weights_map: dict[pd.Timestamp, pd.Series] = {}
    turnover_items: dict[pd.Timestamp, float] = {}
    selections: dict[pd.Timestamp, list[str]] = {}

    weights_map[t0] = w.copy()
    selections[t0] = topn.tolist().copy()

    daily_rets = px.pct_change().fillna(0.0)

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

    for t in rebal_dates:
        # equity evolution for block
        t_ip1 = px[px.index > t_i].index[0] # finding trading day after first day of period bc first day of period is the same as last day of last period
        G = np.exp(L.loc[t_i:t, w.index] - L.loc[t_i, w.index])
        equity.loc[t_ip1:t] = equity[t_i] * (G * w).sum(axis=1).loc[t_ip1:]

        # weights evolution and turnover
        if t != rebal_dates[-1]: # catch strategy end day
            topn = topn_map[t.to_period("M")]
            w_new = pd.Series(1.0 / len(topn), index=topn)
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
            weights_map[t] = w_new.copy()
            selections[t] = topn.tolist().copy()
            turnover_items[t] = turnover
    equity[t0] = 1

    # accumulating data
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
        "rank_col": rank_col,
        "N": N,
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