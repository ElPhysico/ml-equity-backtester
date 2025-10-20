# src/mlbt/calendar.py
"""
Calendar utilities.

This module centralizes calendar-related helpers for research and backtesting,
such as constructing month-end, quarter-end, year-end, or week-end trading
schedules. Functions return canonical grids that downstream components
(strategies, feature builders, label generators) can share to avoid
duplication and alignment errors.
"""
import pandas as pd
import logging

from typing import Literal

Freq = Literal["D", "B", "W", "M"]
N_MAP: dict[Freq, int] = {"D": 252, "B": 261, "W": 52, "M": 12}


def years_spanned(idx: pd.DatetimeIndex) -> float:
    return float((idx[-1] - idx[0]) / pd.Timedelta(days=365.25))

def freq_from_datetimeindex(idx: pd.DatetimeIndex) -> str:
    freq = idx.freqstr
    if freq is None:
        freq = pd.infer_freq(idx)
    if freq is None:
        logging.warning(f"Couldn't infer Datetimeindex period, using 'D'.")
        freq = "D"
    return freq

def ann_factor(freq: Freq, overwrite: int | None = None) -> int:
    return N_MAP[freq] if overwrite is None else overwrite

def tdy_from_index(idx: pd.DatetimeIndex) -> int:
    f = freq_from_datetimeindex(idx)
    return ann_factor(f, overwrite=None)


def build_month_end_grid(
    px_wide: pd.DataFrame
) -> pd.DataFrame:
    """
    Construct a canonical (month, ticker) grid of effective month-end trading dates and closes. This function becomes heavy for large universes.

    Parameters
    ----------
    px_wide : pd.DataFrame
        Wide daily price table with a DatetimeIndex (tz-naive, ascending, unique)
        and columns as tickers. Values are daily closes; missing values are allowed.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by (month: Period['M'], ticker: str) with columns:
        - trade_dt_M : pd.Timestamp
            Actual last trading date observed for that ticker within the month.
        - px_M : float
            Close on trade_dt_M.
    """
    logging.warning("Building month grid (remove this warning at some point, currently warns because we do not want to build month grid unnecessary)")
    # sorting index
    px_wide = px_wide.sort_index()

    # convert to long table
    px_long = px_wide.stack().rename_axis(index=["date", "ticker"]).to_frame("close").sort_index()

    # add month column for convenience
    px_long["month"] = px_long.index.get_level_values("date").to_period("M")

    # now use month to find last trading day
    g = px_long.groupby(["ticker", "month"], observed=True)
    last_rows = g.tail(1).reset_index()
    last_rows = last_rows.rename(columns={
        "date": "trade_dt_M",
        "close": "px_M"
    })
    last_rows = last_rows.sort_values(["ticker", "month"])
    me_table = last_rows.set_index(["month", "ticker"]).sort_index().loc[:, ["trade_dt_M", "px_M"]]

    return me_table