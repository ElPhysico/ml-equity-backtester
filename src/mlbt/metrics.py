# src/mlbt/metrics.py
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import Literal, Optional

from mlbt.strategy_result import StrategyResult

Freq = Literal["D", "W", "M"]

# Annualization factors: daily=252, weekly=52, monthly=12
_N_MAP: dict[Freq, int] = {"D": 252, "W": 52, "M": 12}


@dataclass(frozen=True)
class MetricsResult:
    total_return: float
    cagr: float
    vol_ann: float
    sharpe: float
    max_drawdown: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_series(self, name: Optional[str] = None):
        s = pd.Series(asdict(self))
        if name is not None:
            s.name = name
        return s
    
    def to_string(self) -> str:
        s = f"TR: {100*self.total_return:.2f}%"
        s += f" | Sharpe: {self.sharpe:.2f}"
        s += f" | CAGR: {100*self.cagr:.2f}%"
        s += f" | MaxDD: {100*self.max_drawdown:.2f}%"
        s += f" | ann_vol: {100*self.vol_ann:.2f}%"
        return s



# ---------------- Helpers ----------------

def _validate_index(s: pd.Series) -> pd.Series:
    out = s.copy()

    # 1) Coerce to DatetimeIndex
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="raise")

    # 2) Make tz-naive (convert to UTC then drop tz if tz-aware)
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)

    # 3) Sort
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()

    # 4) Ensure uniqueness
    if not out.index.is_unique:
        raise ValueError("Duplicate timestamps in index.")

    return out


def _as_returns_from_equity(equity: pd.Series) -> pd.Series:
    eq = _validate_index(equity).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
    return eq.pct_change().dropna()

def _as_equity_from_returns(returns: pd.Series) -> pd.Series:
    ret = _validate_index(returns).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
    return (1.0 + ret).cumprod()

def _years_spanned(idx: pd.DatetimeIndex) -> float:
    return float((idx[-1] - idx[0]) / pd.Timedelta(days=365.25))

def _ann_factor(freq: Freq) -> int:
    return _N_MAP[freq]

def _to_periodic_returns(daily_returns: pd.Series, freq: Freq) -> pd.Series:
    ret = _validate_index(daily_returns).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()

    if freq == "D":
        return ret

    # geometric link within each period: (1+r).prod() - 1
    gross = (1.0 + ret)

    if freq == "W":
        # Friday-to-Friday weekly bars (non-overlapping)
        return gross.resample("W-FRI").prod().dropna() - 1.0
    elif freq == "M":
        # Month-end bars (non-overlapping)
        return gross.resample("ME").prod().dropna() - 1.0
    else:
        raise ValueError(f"Unsupported freq '{freq}'. Use 'D', 'W', or 'M'.")
    
def _prepend_equity_baseline(equity: pd.Series, baseline: float = 1.0) -> pd.Series:
    eq = _validate_index(equity).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
    first_ts = eq.index[0]
    prev_ts = first_ts - pd.offsets.BDay(1)
    eq0 = pd.Series([baseline], index=pd.DatetimeIndex([prev_ts], name=equity.index.name))
    eq = pd.concat([eq0, eq]).sort_index()
    return eq


# ---------------- Public metric primitives ----------------

def total_return(
    equity: Optional[pd.Series] = None,
    returns: Optional[pd.Series] = None
) -> float:
    if returns is not None:
        ret = _validate_index(returns).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
        return float((1.0 + ret).cumprod().iloc[-1]) - 1.0
    elif equity is not None:
        eq = _validate_index(equity).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
        return float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    else:
        raise ValueError("Either equity or returns are required to compute total return.")
    
def cagr(
    equity: Optional[pd.Series] = None,
    returns: Optional[pd.Series] = None
) -> float:
    tr = total_return(equity, returns)
    if returns is not None:
        ret = _validate_index(returns).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
        y = _years_spanned(ret.index)
    elif equity is not None:
        eq = _validate_index(equity).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
        y =_years_spanned(eq.index)
    else:
        raise ValueError("Either equity or returns are required to compute CAGR.")
    return float((1 + tr) ** (1/y) - 1.0)

def vol_annualized(
    returns: pd.Series,
    freq: Freq = "D",
    ddof: int = 0
) -> float:
    ret = _validate_index(returns).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
    N = _ann_factor(freq)
    return float(ret.std(ddof=ddof) * N**(1/2))

def sharpe(
    returns: pd.Series,
    freq: Freq = "D",
    rf_ann: float = 0.0,
) -> float:
    ret = _validate_index(returns).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
    N = _ann_factor(freq)
    rf_periodic = (1 + rf_ann)**(1/N) - 1
    mean_excess_ann = (ret.mean() - rf_periodic) * N
    vol_ann = vol_annualized(ret, freq)
    sharpe = mean_excess_ann / vol_ann
    return float(sharpe)
    
def max_drawdown(
    equity: Optional[pd.Series] = None,
    returns: Optional[pd.Series] = None,
) -> float:
    if equity is not None:
        eq = _validate_index(equity).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
        dd = eq / eq.cummax() - 1.0
        return float(-dd.min())
    elif returns is not None:
        ret = _validate_index(returns).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
        eq = _as_equity_from_returns(ret)
        eq_baseline = _prepend_equity_baseline(eq)
        dd = eq_baseline / eq_baseline.cummax() - 1.0
        return float(-dd.min())
    else:
        raise ValueError("Either equity or returns are required to compute max drawdown.")



# ---------------- Orchestrator ----------------

def compute_metrics(
    equity: Optional[pd.Series] = None,
    returns: Optional[pd.Series] = None,
    freq: Freq = "D",
    rf_ann: float = 0
) -> None:
    """
    Compute performance metrics for an equity curve, returns or StrategyResult.

    If a StrategyResult object is provided instead of a Series, its
    `.equity` attribute is used automatically.
    """
    if isinstance(equity, StrategyResult):
        equity = equity.equity
        
    eq, ret = None, None
    if returns is not None:
        ret = _validate_index(returns).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
    elif equity is not None:
        eq = _validate_index(equity).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
    else:
        raise ValueError("Either equity or returns are required to compute metrics.")
    
    if ret is not None:
        ret_for_vol = _to_periodic_returns(ret, freq)
    else:
        daily_ret = _as_returns_from_equity(eq)
        ret_for_vol = _to_periodic_returns(daily_ret, freq)
    vol_ann = vol_annualized(ret_for_vol, freq)

    metrics = MetricsResult(
        total_return=total_return(eq, ret),
        cagr=cagr(eq, ret),
        vol_ann=vol_ann,
        sharpe=sharpe(ret_for_vol, freq, rf_ann),
        max_drawdown=max_drawdown(eq, ret)
    )

    return metrics