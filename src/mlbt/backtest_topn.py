# src/mlbt/backtest_topn.py
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
from typing import Dict, List, Optional
import logging

from mlbt.strategies import StrategyResult
from mlbt.load_prices import flag_suspect_splits


def backtest_topn(
    px_wide: pd.DataFrame,
    predictions: pd.DataFrame,
    rank_col: str,
    N: int = 10,
    cost_bps: float = 5.0,
    name: Optional[str] = None
) -> StrategyResult:
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
        MultiIndex DataFrame with index levels `["month", "ticker"]` containing
        at least the column specified by `rank_col`.  Each row represents the
        model signal or score for a given month and ticker.

    rank_col : str
        Name of the column in `predictions` to rank tickers by.  Higher values
        correspond to stronger buy signals.

    N : int, default=10
        Number of tickers to select each month.

    cost_bps : float, default=5.0
        One-way transaction cost in basis points applied on turnover at each
        rebalance.  A cost of 5.0 corresponds to 0.05 %.

    name : str, optional
        Optional identifier for the resulting strategy.  Defaults to
        `"{rank_col}_top{N}"`.

    Returns
    -------
    StrategyResult
        Object containing the full daily equity curve, rebalance dates,
        portfolio weights, selections, turnover series, and parameter metadata.

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
    # check for potential stocksplits
    flag_suspect_splits(px_wide)

    # aligning calendars
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
    
    # topn map
    topn_tbl = preds.groupby("month", observed=True).head(N)
    counts = topn_tbl.groupby("month", observed=True).size()
    bad_months = counts[counts != N]
    if not bad_months.empty:
        raise ValueError(
            f"{len(bad_months)} months do not have exactly N={N} tickers, "
            f"examples:\n{bad_months.head(3)}"
        )
    topn_map: Dict[pd.Period, pd.Index] = topn_tbl.reset_index().groupby("month", observed=True)["ticker"].apply(pd.Index).to_dict()
    
    # initial portfolio allocation on first rebalance date
    topn = topn_map[t0.to_period("M")]
    w = pd.Series(1.0 / len(topn), index=topn)
    
    # we start at normalized equity 1.0
    eq = 1.0
    equity = [eq] # construct daily equity curve
    charged_init = False # initial investment fee not charged yet
    weights_map: Dict[pd.Timestamp, pd.Series] = {}
    turnover_items: Dict[pd.Timestamp, float] = {}
    selections: Dict[pd.Timestamp, List[str]] = {}

    weights_map[t0] = w.copy()
    selections[t0] = topn.tolist().copy()

    px_alloc = px.loc[t0:] # investment horizon
    daily_rets = px_alloc.pct_change().iloc[1:] # first pct change is nan

    # check for unusual large number of nans
    row_month = daily_rets.index.to_period("M")
    nan_share_df = daily_rets.isna().groupby(row_month).mean()
    bad_mask = nan_share_df > 0.3
    if bad_mask.to_numpy().any():
        bad_preview = nan_share_df.where(bad_mask).stack().sort_values(ascending=False).head(10)
        logging.warning(
            f"High NaN-share detected: {bad_mask.to_numpy().sum()} (month,ticker) pairs "
            f"with >30% missing daily returns.\nPreview:\n{bad_preview}"
        )

    # run backtest
    for t in px_alloc.index[1:]:
        # daily equity and weights update
        r = daily_rets.loc[t, w.index].fillna(0.0)
        R_t = float((w * r).sum())
        eq *= (1.0 + R_t)
        w = (w * (1.0 + r)) / (1.0 + R_t)

        # rebalance dates
        if t in rebal_dates[1:-1]: # no rebal on last rebal date
            topn = topn_map[t.to_period("M")]
            w_new = pd.Series(1.0 / len(topn), index=topn)
            union = w.index.union(w_new.index)
            w_old_aligned = w.reindex(union).fillna(0.0).astype("float64")
            w_new_aligned = w_new.reindex(union).fillna(0.0).astype("float64")
            turnover = (w_new_aligned - w_old_aligned).abs().sum()
            cost_frac = (cost_bps / 1e4) * turnover
            eq *= (1.0 - cost_frac)
            w = w_new
            
            # tracking
            weights_map[t] = w_new.copy()
            selections[t] = topn.tolist().copy()
            turnover_items[t] = turnover
        
        if not charged_init:
            # the very first transaction cost is included in first day return
            eq *= (1.0 - cost_bps / 1e4) # initial cost is always on the total notional
            charged_init = True

        equity.append(eq)
    
    if name is None: # highly recommend passing a name for identification
        name = f"{rank_col}_top{N}"
    eq = pd.Series(equity, index=px_alloc.index, name=name)

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

    params = {
        "N": N,
        "cost_bps": cost_bps
    }

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
    
    return res