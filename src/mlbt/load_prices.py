# src/mlbt/load_prices.py
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Optional, Sequence, Union


def load_prices(
    in_dir: Union[str, Path],
    tickers: Sequence[str],
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None
) -> pd.DataFrame:
    """
    Returns wide price table: index=date, columns=tickers, values=close.
    """
    in_dir = Path(in_dir)
    filters = [('ticker','in', tickers)]
    if start: filters.append(('date','>=', pd.to_datetime(start)))
    if end:   filters.append(('date','<=', pd.to_datetime(end)))
    df = pd.read_parquet(in_dir, engine="pyarrow", columns=["date", "ticker", "close"], filters=filters)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    else:
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    px = df.pivot_table(index=df.index, columns="ticker", values="close", observed=False)
    px = px.sort_index()
    return px

def flag_suspect_splits(
    prices: pd.DataFrame,
    min_change: float = -0.50,
    max_change: float =  1.00
) -> pd.DataFrame:
    """
    Scan a wide price table (index=date, columns=tickers, values=close) for jumpy moves that might indicate splits or bad data.

    Returns a tidy DataFrame with rows for suspicious (ticker, date) events and columns:
        ['ticker','date','close_prev','close_curr','pct_change','ratio','reason']

    Notes
    - Uses daily close-to-close percentage change.
    - By default flags moves outside [-50%, +100%].
    """
    px = prices.sort_index()
    px_prev = px.shift(1)

    pct = (px / px_prev) - 1.0
    ratio = 1.0 + pct

    df = pd.DataFrame({
        "pct_change": pct.stack(future_stack=True),
        "ratio": ratio.stack(future_stack=True),
        "close_curr": px.stack(future_stack=True),
        "close_prev": px_prev.stack(future_stack=True),
    }).reset_index()
    df.columns = ["date", "ticker", "pct_change", "ratio", "close_curr", "close_prev"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["pct_change", "ratio", "close_prev", "close_curr"])

    flagged = (df["pct_change"] < min_change) | (df["pct_change"] > max_change)
    reasons = np.where(flagged, "range_exceed", "")


    out = df.loc[flagged, ["ticker", "date", "close_prev", "close_curr", "pct_change", "ratio"]].copy()
    out["reason"] = reasons[np.where(flagged)[0]]
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    if not out.empty:
        for tkr, grp in out.groupby("ticker", observed=False):
            first_rows = grp.head(3)
            if first_rows.empty:
                continue
            examples = ", ".join(f"{r.date.date()} ({r.pct_change:+.1%})" for r in first_rows.itertuples())
            extra = "" if len(grp) <= 3 else f" … and {len(grp) - 3} more"
            logging.warning(f"[split-check] {tkr}: {len(grp)} suspicious jump(s). e.g., {examples}{extra}")

    return out