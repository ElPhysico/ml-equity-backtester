# src/mlbt/strategies/momentum_m_skip.py
"""
Cross-sectional momentum: M-month lookback, skip S most-recent months (e.g., 12-1).
Provides a reusable signal builder and a simple top-N equal-weight strategy runner.
"""
from typing import Optional
import pandas as pd
import numpy as np
from math import ceil


def compute_momentum_m_skip_signal(
    px: pd.DataFrame,
    *,
    lookback_months: int = 12,
    skip_months: int = 1,
    min_hist_ratio: float = 0.95,
    use_log: bool = True,
) -> pd.DataFrame:
    """
    Build month-end cross-sectional momentum signals using an M-month lookback that skips the most recent S months (e.g., 12-1 momentum).

    Parameters
    ----------
    px : pd.DataFrame
        Daily close prices. Index: DatetimeIndex (tz-naive), ascending unique.
        Columns: ticker symbols.
    lookback_months : int, default 12
        Number of months in the lookback window used for the signal.
    skip_months : int, default 1
        Number of most-recent months to skip (to avoid 1-month reversal).
    min_hist_ratio : float, default 0.95
        Minimum fraction of expected observations in the lookback window required for a ticker to receive a valid signal at a given month-end.
    use_log : bool, default True
        If True, aggregate log returns for the signal; otherwise aggregate simple returns.

    Returns
    -------
    pd.DataFrame
        Month-end indexed DataFrame of momentum signals (rows = month-ends,
        columns = tickers). NaNs where ineligible due to insufficient history.

    Notes
    -----
    - This function only constructs the signal; it does not form portfolios.
    - The first valid signal date will be after at least `lookback_months + skip_months`
      months of history are available.
    """
    # prepare data and filter for bad columns
    me_closes = px.dropna(axis=1, how="all").ffill().groupby(px.index.to_period('M')).last()
    first = me_closes.iloc[0:1]
    tickers = first.columns[first.notna().iloc[0]]
    me_closes = me_closes.loc[:, tickers]

    # simple returns
    ret = me_closes.pct_change().dropna()

    W = lookback_months - skip_months
    K = ceil(min_hist_ratio * W)

    # shift returns to respect skipping months
    ret_shift = ret.shift(skip_months)

    # create signal using sum(log(1+r)) over rolling window
    signal_log = np.log1p(ret_shift).rolling(W, min_periods=1).sum()

    # check if enough data for eligibility
    obs = ret_shift.rolling(W).count()
    signal_log = signal_log.where(obs >= K)

    # create log or simple signal
    signal = signal_log if use_log else np.expm1(signal_log)
    signal = signal.rename_axis(index="month_end", columns="ticker")

    return signal


def cross_sectional_momentum_topn(
    px: pd.DataFrame,
    *,
    top_n: int = 10,
    cost_bps: float = 0.0,
    lookback_months: int = 12,
    skip_months: int = 1,
    min_hist_ratio: float = 0.95,
    name: Optional[str] = None,
) -> pd.Series:
    """
    Run a simple cross-sectional momentum strategy:
    - At each month-end, rank tickers by the M-skip-S momentum signal.
    - Select the top-N tickers, equal-weight them, and hold for one month.
    - Apply one-way bps on turnover each rebalance.
    - Output a monthly equity curve (post-cost).

    Parameters
    ----------
    px : pd.DataFrame
        Daily close prices. Index: DatetimeIndex (tz-naive), ascending unique.
        Columns: ticker symbols.
    top_n : int, default 10
        Number of names to hold each month (equal-weight).
    cost_bps : float, default 0.0
        One-way transaction cost in basis points applied on turnover each rebalance.
    lookback_months : int, default 12
        Lookback window length (months) for the momentum signal.
    skip_months : int, default 1
        Number of most-recent months to skip in the signal.
    min_hist_ratio : float, default 0.95
        Minimum data availability in the signal window for eligibility.
    name : str, optional
        Name for the returned equity curve Series. If None, defaults to
        f"MOM_{lookback_months}-{skip_months}_N{top_n}".

    Returns
    -------
    pd.Series
        Monthly equity curve (post-cost), index = month-ends, name as above.

    Notes
    -----
    - Uses equal weights among selected names (no score-weighting).
    - The first tradable month occurs after a valid signal can be computed.
    - Turnover is computed as 0.5 * sum(|w_new - w_old|); first invested month
      incurs full entry cost.
    """
    # create trading signal
    signal = compute_momentum_m_skip_signal(
        px=px,
        lookback_months=lookback_months,
        skip_months=skip_months,
        min_hist_ratio=min_hist_ratio,
        use_log=True
    )
    # find first month with signals and align calendar
    first_month = signal.first_valid_index()
    px = px.dropna(axis=1, how="all").ffill()
    px = px[px.index.to_period("M") >= first_month]

    # get rebalance dates
    s = pd.Series(px.index, index=px.index)
    rebal_dates = pd.DatetimeIndex(s.groupby(px.index.to_period('M')).last().values)
    t0 = rebal_dates[0] # first trading day where we allocate
    
    # compute initial portfolio allocation
    this_signal = signal[signal.index == t0.to_period("M")].iloc[0].dropna()
    topn = this_signal.nlargest(top_n)
    w = pd.Series(1.0 / len(topn), index=topn.index)

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
        if date in rebal_dates[1:-1]: # no rebalance on last rebalance date
            this_signal = signal[signal.index == (date.to_period("M"))].iloc[0].dropna()
            topn = this_signal.nlargest(top_n)
            w_new = pd.Series(1.0 / len(topn), index=topn.index)
            turnover = float((w_new - w).abs().sum())
            cost_frac = (cost_bps / 1e4) * turnover
            eq *= (1.0 - cost_frac)
            w = w_new
        # elif date == rebal_dates[-1]:
        #     # end of portfolio, we sell everything
            # eq *= (1.0 - cost_bps / 1e4)
        if not charged_init:
            # the very first transaction cost is included in first day return
            eq *= (1.0 - cost_bps / 1e4) # initial cost is always on the total notional
            charged_init = True

        equity.append(eq)

    if name is None:
        name = f"MOM_{lookback_months}-{skip_months}_N{top_n}"
    eq = pd.Series(equity, index=px.index, name=name)

    return eq

    
