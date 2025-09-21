# src/mlbt/strategies/buy_and_hold.py
import pandas as pd


def buy_and_hold(px: pd.DataFrame, cost_bps: float = 0.0) -> pd.Series:
    """
    Equal-weight buy-and-hold:
    - On the first valid date (t0), identify tickers with prices at t0.
    - Allocate 1/N across those tickers and hold weights unchanged to the end.
    - Apply one-way cost_bps once at entry (embedded via first-day adjustment).

    Parameters
    ----------
    px : pd.DataFrame
        Daily close prices. Index: DatetimeIndex (tz-naive), ascending unique.
        Columns: ticker symbols. May contain NaNs; tickers without a price at t0
        are excluded from the initial portfolio.
    cost_bps : float, default 0.0
        One-way transaction cost in basis points charged once at portfolio entry.

    Returns
    -------
    pd.Series
        Daily equity curve (post-cost), index = trading dates from t0 onward,
        name = 'equity'.

    Notes
    -----
    - Weights are fixed after t0 (no rebalancing).
    - Entry cost is applied by scaling the equity on day 1 and resetting eq[t0] = 1.0,
      which effectively reduces the first day’s return by cost_bps.
    - Input is expected to be price-level data; internal normalization uses px / px.loc[t0].
    - Asserts at least one ticker has a finite price at t0.
    """
    first = px.dropna(axis=1, how="all").iloc[0:1]
    t0 = first.index[0]
    tickers = first.columns[first.notna().iloc[0]]
    assert len(tickers) > 0, "No tickers with data on t0"

    w0 = pd.Series(1.0 / len(tickers), index=tickers, dtype="float64")

    norm = (px / px.loc[t0]).loc[t0:]
    norm = norm.reindex(columns=tickers).ffill()

    eq = (1.0 - cost_bps / 1e4) * (norm * w0).sum(axis=1)
    # the below effectively changes the return of the first day by the buy cost
    eq[t0] = 1.0
    eq.name = "equity"

    return eq
