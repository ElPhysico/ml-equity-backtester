# src/mlbt/utils.py
import pandas as pd
import hashlib
from pathlib import Path
from datetime import datetime, UTC
import os
import json
from typing import Any, Dict, List, Optional, Tuple

PROJECT_SENTINELS = ("pyproject.toml", ".git")

def sha1_of_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def short_sha1_of_json(obj: dict, length: int = 8) -> str:
    """Deterministic short hash for run IDs (stable across Python processes)."""
    return sha1_of_str(json.dumps(obj, sort_keys=True, separators=(",", ":")))[:length]

def utc_now_iso() -> str:
    # ISO 8601 with 'Z' suffix
    return datetime.now(UTC).replace(microsecond=0).isoformat() + "Z"

def to_iso_date(dt: pd.Timestamp) -> str:
    """YYYY-MM-DD (no time)."""
    if isinstance(dt, pd.Timestamp):
        return dt.date().isoformat()
    # fallback for plain date/datetime
    try:
        return pd.Timestamp(dt).date().isoformat()
    except Exception:
        return str(dt)

def find_project_root(start: Path | None = None) -> Path:
    """
    Resolve the project root by:
      1) MLBT_ROOT env var, if set
      2) walking upward from `start` (or this file) until a sentinel is found
      3) fallback to current working directory
    """
    env = os.environ.get("MLBT_ROOT")
    if env:
        return Path(env).resolve()

    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        for s in PROJECT_SENTINELS:
            if (parent / s).exists():
                return parent
    return Path.cwd().resolve()

def universe_id_from_grid(month_grid: pd.DataFrame) -> str:
    # Expect index = (month, ticker)
    tickers = sorted(set(month_grid.index.get_level_values("ticker")))
    uh = sha1_of_str(",".join(tickers))
    return f"u{len(tickers)}_{uh[:6]}"

def coverage_from_grid(month_grid: pd.DataFrame) -> Tuple[str, str]:
    months = month_grid.index.get_level_values("month")
    start = str(months.min())  # "YYYY-MM"
    end   = str(months.max())  # "YYYY-MM"
    return start, end

def config_hash(universe_hash: str, params: Dict[str, Any], feature_names: list[str]) -> str:
    # Stable, readable-ish
    key = {
        "universe_hash": universe_hash,
        "params": params,
        "features": feature_names,
    }
    return sha1_of_str(json.dumps(key, sort_keys=True))

def tickers_from_grid(month_grid: pd.DataFrame) -> List[str]:
    """
    Return sorted unique tickers from a (month, ticker) MultiIndex grid.
    """
    return sorted(set(month_grid.index.get_level_values("ticker")))


def universe_hash_from_grid(month_grid: pd.DataFrame) -> str:
    """
    Stable SHA-1 hash over the sorted ticker list of the grid universe.
    """
    tk = tickers_from_grid(month_grid)
    return sha1_of_str(",".join(tk))


def universe_summary_from_grid(month_grid: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a compact universe summary dict from a grid.
    """
    tickers = tickers_from_grid(month_grid)
    return {
        "tickers": tickers,
        "count": len(tickers),
        "hash": sha1_of_str(",".join(tickers)),
        "id": universe_id_from_grid(month_grid),
    }


# ------------------------------------------------

def build_panel_meta(
    month_grid: pd.DataFrame,
    *,
    feature_names: List[str],
    params: Dict[str, Any],
    panel_name: str,
    panel_version: str = "v0",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Construct a standardized metadata dict for any monthly panel.

    Parameters
    ----------
    month_grid : pd.DataFrame
        MultiIndex (month, ticker) grid controlling coverage/universe.
    feature_names : list of str
        Feature column names this panel provides.
    params : dict
        Parameterization used to build the panel (windows, flags, etc.).
    panel_name : str
        Human-readable name, e.g. "feature_panel" or "label_panel".
    panel_version : str, default "v0"
        Version tag of the panel implementation.
    extra : dict, optional
        Free-form additions (e.g., data source info), merged at top level.

    Returns
    -------
    meta : dict
        Standardized metadata including created_at, coverage, universe,
        panel identifier, features, params, and a config_hash.
    """
    start_month, end_month = coverage_from_grid(month_grid)
    univ = universe_summary_from_grid(month_grid)
    cfg_hash = config_hash(univ["hash"], params, feature_names)

    meta: Dict[str, Any] = {
        "created_at": utc_now_iso(),
        "panel": {"name": panel_name, "version": panel_version},
        "data_coverage": {"start": start_month, "end": end_month},
        "universe": univ,
        "features": feature_names,
        "params": params,
        "config_hash": cfg_hash,
    }
    if extra:
        # Shallow merge; caller controls collisions intentionally.
        meta.update(extra)
    return meta


def build_run_meta(
    predictions: pd.DataFrame,
    res: Any,
    *,
    run_name: Optional[str],
    runner_name: str,
    runner_version: str = "v0",
    params: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Construct a standardized metadata dictionary for any backtest or experiment run.

    Parameters
    ----------
    predictions : pandas.DataFrame
        MultiIndex (month, ticker) table used to infer coverage and universe summary.
    res : Any
        StrategyResult-like object with `.rebal_dates` and `.name`; adds brief strategy info.
    run_name : str, optional
        Friendly label chosen by the user (appears in logs and filenames).
    runner_name : str
        Name of the orchestration function (e.g., "topn_from_predictions").
    runner_version : str, default "v0"
        Implementation version tag.
    params : dict
        Parameters governing the run (e.g., rank_col, N, cost_bps, strict, model_cfg, etc.).
    metrics : dict, optional
        Optional metrics summary (Sharpe, CAGR, etc.) for inclusion in the metadata.
    extra : dict, optional
        Free-form additional fields merged at top level.

    Returns
    -------
    meta : dict
        Standardized run metadata including created_at, runner info,
        coverage, universe preview, parameters, optional metrics,
        and optional strategy summary.
    """
    start_month, end_month = coverage_from_grid(predictions)
    univ = universe_summary_from_grid(predictions)

    strat_info = None
    try:
        strat_info = {
            "strategy_name": getattr(res, "name", None),
            "n_rebalances": int(len(res.rebal_dates)) if getattr(res, "rebal_dates", None) is not None else 0,
            "first_rebalance": to_iso_date(res.rebal_dates[0]) if getattr(res, "rebal_dates", None) is not None and len(res.rebal_dates) else None,
            "last_rebalance": to_iso_date(res.rebal_dates[-1]) if getattr(res, "rebal_dates", None) is not None and len(res.rebal_dates) else None,
        }
    except Exception:
        strat_info = None

    meta: Dict[str, Any] = {
        "created_at": utc_now_iso(),
        "runner": {"name": runner_name, "version": runner_version},
        "run_name": run_name,
        "data_coverage": {"start": start_month, "end": end_month},
        "universe": univ,
        "params": params,
    }
    if strat_info:
        meta["strategy"] = strat_info
    if metrics:
        meta["metrics"] = metrics
    if extra:
        meta.update(extra)

    return meta




