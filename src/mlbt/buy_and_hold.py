# src/mlbt/buy_and_hold.py
import pandas as pd


def buy_and_hold(px: pd.DataFrame, cost_bps: float = 0.0) -> pd.Series:
    """
    Buy-and-hold strategy: equal weight on first valid day with enough tickers, hold to end.
    Charges cost_bps (basis points) once at start.

    Returns equity series (index=date, values=equity).
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
