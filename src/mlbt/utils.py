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

def utc_now_iso() -> str:
    # ISO 8601 with 'Z' suffix
    return datetime.now(UTC).replace(microsecond=0).isoformat() + "Z"

def _to_iso_date(dt: pd.Timestamp) -> str:
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

def _short_sha1_of_json(obj: dict, length: int = 8) -> str:
    """Deterministic short hash for run IDs (stable across Python processes)."""
    return sha1_of_str(json.dumps(obj, sort_keys=True, separators=(",", ":")))[:length]

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

def _short_sha1_of_json(obj: dict, length: int = 8) -> str:
    """Deterministic short hash for run IDs (stable across Python processes)."""
    return sha1_of_str(json.dumps(obj, sort_keys=True, separators=(",", ":")))[:length]


def _to_iso_date(dt: pd.Timestamp) -> str:
    """YYYY-MM-DD (no time)."""
    if isinstance(dt, pd.Timestamp):
        return dt.date().isoformat()
    # fallback for plain date/datetime
    try:
        return pd.Timestamp(dt).date().isoformat()
    except Exception:
        return str(dt)


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



def save_artifacts(
    *,
    run_name: Optional[str],
    params: Dict[str, Any],
    res: Any,  # duck-typed StrategyResult (expects .name, .equity, .weights, .turnover, .selections, .rebal_dates)
    preds: Optional[pd.DataFrame] = None,
    metrics: Optional[Dict[str, Any]] = None,
    out_dir: Path | str = "outputs/backtests",
    code_paths: Optional[List[Path]] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Persist core backtest artifacts and return run directory + metadata.

    Parameters
    ----------
    run_name : str, optional
        Friendly label for this run (stored in metadata).
    params : dict
        Parameters relevant to the run (e.g., N, cost_bps, rank_col, strict, feature/model cfg).
    res : Any
        Strategy-like result with attributes:
          - name : str
          - equity : pd.Series (index=date)
          - weights : pd.DataFrame (index=rebal_date, columns=tickers)
          - turnover : pd.Series (index=rebal_date)
          - selections : Dict[pd.Timestamp, List[str]]
          - rebal_dates : pd.DatetimeIndex
    preds : pd.DataFrame, optional
        Predictions table to persist (e.g., (month, ticker) → y_pred[, y_true]).
    metrics : dict, optional
        Metrics to persist (e.g., from mlbt.metrics.compute_metrics).
    out_dir : Path | str, default "outputs/backtests"
        Root folder under the project where run subfolder is created.
    code_paths : List[Path], optional
        List of source files to hash and record in metadata.

    Returns
    -------
    run_dir : Path
        Path to the created run directory.
    run_meta : dict
        Metadata also written to run_meta.json (contains paths to saved files).
    """
    # Resolve paths
    root = find_project_root()
    out_root = (root / Path(out_dir)).resolve()

    # Create run_id (timestamp + short hash of params)
    created_at = utc_now_iso()
    stamp = created_at.replace("-", "").replace(":", "").replace("Z", "").replace("T", "-")
    short_hash = _short_sha1_of_json(params, length=8)
    run_id = f"{stamp}_{short_hash}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # File paths
    p_equity = run_dir / "equity.csv"
    p_weights = run_dir / "weights.csv"
    p_turnover = run_dir / "turnover.csv"
    p_selections = run_dir / "selections.json"
    p_preds = run_dir / "predictions.parquet" if preds is not None else None
    p_metrics = run_dir / "metrics.json" if metrics is not None else None
    p_meta = run_dir / "run_meta.json"

    # Write core artifacts
    # equity as single-column CSV for easy plotting
    res.equity.to_frame(name=getattr(res, "name", "strategy")).to_csv(p_equity, index=True)
    # weights and turnover
    if getattr(res, "weights", None) is not None:
        res.weights.to_csv(p_weights, index=True)
    if getattr(res, "turnover", None) is not None:
        res.turnover.to_csv(p_turnover, index=True)
    # selections: map ISO date -> list of tickers
    if getattr(res, "selections", None) is not None:
        selections_iso = { _to_iso_date(k): v for k, v in res.selections.items() }
        p_selections.write_text(json.dumps(selections_iso, indent=2))
    # preds & metrics
    if preds is not None:
        preds.to_parquet(p_preds, index=True)
    if metrics is not None:
        p_metrics.write_text(json.dumps(metrics, indent=2))

    # Optional code hashes
    code_hashes: Dict[str, str] = {}
    if code_paths:
        for path in code_paths:
            try:
                data = Path(path).read_bytes()
                code_hashes[str(Path(path))] = hashlib.sha1(data).hexdigest()
            except Exception:
                code_hashes[str(Path(path))] = "<unavailable>"

    # Compose metadata
    run_meta: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": created_at,
        "run_name": run_name,
        "strategy_name": getattr(res, "name", None),
        "paths": {
            "equity_csv": str(p_equity.relative_to(root)),
            "weights_csv": str(p_weights.relative_to(root)) if p_weights.exists() else None,
            "turnover_csv": str(p_turnover.relative_to(root)) if p_turnover.exists() else None,
            "selections_json": str(p_selections.relative_to(root)) if p_selections.exists() else None,
            "predictions_parquet": str(p_preds.relative_to(root)) if p_preds else None,
            "metrics_json": str(p_metrics.relative_to(root)) if p_metrics else None,
            "meta_json": str(p_meta.relative_to(root)),
            "run_dir": str(run_dir.relative_to(root)),
        },
        "params": params,
        "rebalancing": {
            "first_rebalance": _to_iso_date(res.rebal_dates[0]) if getattr(res, "rebal_dates", None) is not None and len(res.rebal_dates) else None,
            "last_rebalance": _to_iso_date(res.rebal_dates[-1]) if getattr(res, "rebal_dates", None) is not None and len(res.rebal_dates) else None,
            "n_rebalances": int(len(res.rebal_dates)) if getattr(res, "rebal_dates", None) is not None else 0,
        },
        "code_hashes": code_hashes or None,
    }

    # Persist metadata
    p_meta.write_text(json.dumps(run_meta, indent=2))

    return run_dir, run_meta
