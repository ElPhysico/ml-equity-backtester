# src/mlbt/specs/metrics.py
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

from mlbt.specs.strategy_result import StrategyResult
from mlbt.calendar import Freq, years_spanned, freq_from_datetimeindex, ann_factor


@dataclass(frozen=True)
class MetricsResult:
    total_return: float
    cagr: float             # = geometric mean
    arithmetic_mean: float
    vol_ann: float
    sharpe: float
    max_drawdown: float
    freq: Freq
    ppy: float              # periods per year
    ann_turnover: float | None = None # added when calling StrategyResult.compute_metrics()

    def to_dict(self) -> dict:
        return asdict(self)

    def to_series(self, name: str | None = None):
        s = pd.Series(asdict(self))
        if name is not None:
            s.name = name
        return s
    
    def to_string(self) -> str:
        s = f"TR: {self.total_return:.2%}"
        s += f" | Sharpe: {self.sharpe:.2f}"
        s += f" | CAGR: {self.cagr:.2%}"
        s += f" | Arithmetic mean: {self.arithmetic_mean:.2%}"
        s += f" | MaxDD: {self.max_drawdown:.2%}"
        s += f" | Ann. Vol: {self.vol_ann:.2%}"
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

def _to_periodic_returns(daily_returns: pd.Series, freq: Freq) -> pd.Series:
    ret = _validate_index(daily_returns).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()

    if freq == "D" or freq == "B":
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
        raise ValueError(f"Unsupported freq '{freq}'. Use 'D', 'B', 'W', or 'M'.")


# ---------------- Public metric primitives ----------------

def total_return(
    equity: pd.Series | None = None,
    returns: pd.Series | None = None
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
    equity: pd.Series | None = None,
    returns: pd.Series | None = None
) -> float:
    tr = total_return(equity, returns)
    if returns is not None:
        ret = _validate_index(returns).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
        y = years_spanned(ret.index)
    elif equity is not None:
        eq = _validate_index(equity).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
        y = years_spanned(eq.index)
    else:
        raise ValueError("Either equity or returns are required to compute CAGR.")
    return float((1 + tr) ** (1/y) - 1.0)

def arithmetic_mean(
    equity: pd.Series | None = None,
    returns: pd.Series | None = None,
    freq: Freq = "D",
    overwrite: int | None = None # use to overwrite ann_factor/periods per year
) -> float:
    if returns is not None:
        ret = _validate_index(returns).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
    elif equity is not None:
        eq = _validate_index(equity).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
        ret = eq.pct_change().dropna()
    else:
        raise ValueError("Either equity or returns are required to compute arithmetic mean.")
    mean_ret = ret.mean()
    N = ann_factor(freq, overwrite)
    return float(mean_ret * N)    

def vol_annualized(
    returns: pd.Series,
    freq: Freq = "D",
    overwrite: int | None = None, # use to overwrite ann_factor/periods per year
    ddof: int = 0
) -> float:
    ret = _validate_index(returns).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
    N = ann_factor(freq, overwrite)
    return float(ret.std(ddof=ddof) * N**(1/2))

def sharpe(
    returns: pd.Series,
    freq: Freq = "D",
    overwrite: int | None = None, # use to overwrite ann_factor/periods per year
    rf_ann: float = 0.0,
) -> float:
    ret = _validate_index(returns).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
    N = ann_factor(freq, overwrite)
    rf_periodic = (1 + rf_ann)**(1/N) - 1
    mean_excess_ann = (ret.mean() - rf_periodic) * N
    vol_ann = vol_annualized(ret, freq)
    sharpe = mean_excess_ann / vol_ann
    return float(sharpe)
    
def max_drawdown(
    equity: pd.Series | None = None,
    returns: pd.Series | None = None,
) -> float:
    if equity is not None:
        eq = _validate_index(equity).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
        dd = eq / eq.cummax() - 1.0
        return float(-dd.min())
    elif returns is not None:
        ret = _validate_index(returns).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
        eq = _as_equity_from_returns(ret)
        dd = eq / eq.cummax() - 1.0
        return float(-dd.min())
    else:
        raise ValueError("Either equity or returns are required to compute max drawdown.")



# ---------------- Orchestrator ----------------

def compute_metrics(
    equity: pd.Series | StrategyResult | None = None,
    returns: pd.Series | None = None,
    # freq: Freq = "D",
    overwrite: int | None = None, # use to overwrite ann_factor/periods per year
    rf_ann: float = 0
) -> None:
    """
    Compute performance metrics for an equity curve, returns or StrategyResult.

    If a StrategyResult object is provided instead of a Series, its
    `.equity` attribute is used automatically.
    """
    from_sr = {}
    if isinstance(equity, StrategyResult):
        from_sr["ann_turnover"] = equity.ann_turnover
        equity = equity.equity
        
    eq, ret = None, None
    if returns is not None:
        ret = _validate_index(returns).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
        freq = freq_from_datetimeindex(ret.index)
    elif equity is not None:
        eq = _validate_index(equity).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
        freq = freq_from_datetimeindex(eq.index)
    else:
        raise ValueError("Either equity or returns are required to compute metrics.")
    
    
    if ret is not None:
        ret_for_vol = _to_periodic_returns(ret, freq)
    else:
        daily_ret = _as_returns_from_equity(eq)
        ret_for_vol = _to_periodic_returns(daily_ret, freq)
    vol_ann = vol_annualized(ret_for_vol, freq, overwrite)

    metrics = MetricsResult(
        total_return=total_return(eq, ret),
        cagr=cagr(eq, ret),
        arithmetic_mean=arithmetic_mean(eq, ret, freq, overwrite),
        vol_ann=vol_ann,
        sharpe=sharpe(ret_for_vol, freq, overwrite, rf_ann),
        max_drawdown=max_drawdown(eq, ret),
        freq=freq,
        ppy=ann_factor(freq, overwrite),
        **from_sr
    )

    return metrics