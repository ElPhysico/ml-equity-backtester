# src/mlbt/features.py
"""
Feature engineering utilities.

This module houses feature builders for monthly, weekly, or daily research signals.
Functions follow a common pattern:
- Inputs:
  * px (wide daily prices; DatetimeIndex, columns=tickers)
  * features (MultiIndex (month, ticker) table, typically from calendar.build_month_end_grid)
- Behavior:
  * Do not mutate inputs; return a new DataFrame with appended feature columns.
  * No forward-fill inside feature functions; handle missing data via eligibility (NaNs).
- Naming conventions:
  * Monthly features end with `_1m` (e.g., `vol_1m`).
  * Annualized variants are prefixed with `ann_` (e.g., `ann_vol_1m`).
"""
import pandas as pd
import numpy as np


def mom_P_skip_log_1m(
    features: pd.DataFrame,
    P: int = 12,
    skip: int = 1
) -> pd.DataFrame:
    """
    Compute log-return momentum over a P-month lookback window, skipping the most recent `skip` months.

    Parameters
    ----------
    features : pd.DataFrame
        MultiIndex (month, ticker) table that includes 'px_M' (close on month-end).
        Typically from build_month_end_grid.
    P : int, default 12
        Total lookback length in months.
    skip : int, default 1
        Number of most recent months to exclude from the window (e.g. skip=1
        means end the lookback at M-1).

    Returns
    -------
    pd.DataFrame
        Copy of `features` with one additional column:
        - f"mom_{P}_{skip}_log_1m" : log(px_{M-skip} / px_{M-P})
    """
    desc = f"mom_{P}_{skip}_log_1m"

    # get two price series per ticker
    gb = features.groupby("ticker", observed=True)
    px_skip = gb["px_M"].shift(skip)
    px_P = gb["px_M"].shift(P)

    # calculate log returns
    log_return = np.log(px_skip) - np.log(px_P)
    log_return.name = desc

    # safely join log returns into features
    features = features.join(log_return.to_frame(desc), how="left")

    return features


def vol_1m(
    px_wide: pd.DataFrame,
    features: pd.DataFrame,
    min_obs: int = 10,
    annualize: bool = False,
    ddof: int  = 0
) -> pd.DataFrame:
    """
    Compute within-month standard deviation of daily simple returns per ticker.

    Parameters
    ----------
    px_wide : pd.DataFrame
        Wide daily close prices with a DatetimeIndex (tz-naive, ascending, unique)
        and columns as tickers. No forward-fill is applied here.
    features : pd.DataFrame
        A MultiIndex (month: Period['M'], ticker: str) table to which the feature
        will be appended (left join). Typically produced by calendar.build_month_end_grid.
    min_obs : int, default 10
        Minimum number of daily return observations required within the month.
        Months with fewer observations are set to NaN.
    annualize : bool, default False
        If True, multiply by sqrt(252) and write the column as 'ann_vol_1m'.
        If False, write the raw within-month stdev as 'vol_1m'.
    ddof : int, default 0
        Delta degrees of freedom for the standard deviation (0=population, 1=sample).
        Use a value consistent with your metrics module.

    Returns
    -------
    pd.DataFrame
        A copy of `features` with one additional column:
        - 'vol_1m' or 'ann_vol_1m' depending on `annualize`.
    """
    # convert to long dataframe
    px_long = px_wide.stack().rename_axis(index=["date", "ticker"]).to_frame("close").sort_index()
    px_long["month"] = px_long.index.get_level_values("date").to_period("M")

    # calculate returns
    rets = px_long.groupby("ticker", observed=True)["close"].pct_change()

    # calculate std
    std  = rets.groupby([px_long["month"], px_long.index.get_level_values("ticker")], observed=True).std(ddof=ddof)
    std.name = "std"

    # calculate counts and cast to Nan if not enough data in month
    cts = rets.groupby([px_long["month"], px_long.index.get_level_values("ticker")], observed=True).count()
    std = std.mask(cts < min_obs)

    # safely joining vol into features
    if annualize:
        std = std * np.sqrt(252)
        features = features.join(std.to_frame("ann_vol_1m"), how="left")
    else:
        features = features.join(std.to_frame("vol_1m"), how="left")

    return features    