# src/mlbt/io.py
"""
Input / Output utilities.

This module centralizes all read / write helpers for the project.

Sections
--------
1. Feature panels
   - resolve_features_output_dir
   - save_panel
   - maybe_load_panel
2. Label panels
   - resolve_labels_output_dir
   - save_labels
   - maybe_load_labels
3. YAML configs
   - read_yaml
   - write_yaml

Design principles
-----------------
- All save_*() functions create parent directories as needed.
- All maybe_load_*() functions are read-only and never create paths.
- Paths are resolved relative to the project root (via utils.find_project_root)
  unless a custom base_out_dir is provided.
- File formats:
  * Tabular data → Parquet (.parquet)
  * Metadata → JSON (.json)
  * Configs → YAML (.yaml)
"""
from pathlib import Path
import json
import yaml
import pandas as pd
from typing import Dict, Any, Tuple, Optional, Union

from mlbt.utils import find_project_root, short_sha1_of_json, to_iso_date, utc_now_iso


# features io
def resolve_features_output_dir(
    universe_id: str,
    start_month: str,  # "YYYY-MM"
    end_month: str,    # "YYYY-MM"
    config_hash: str,
    base_out_dir: Optional[Path] = None
) -> Path:
    """
    Build a deterministic path where the panel artifacts live.
    Layout: outputs/feature_panel/<universe_id>/start=YYYY-MM__end=YYYY-MM/<config_hash>/
    """
    root = base_out_dir or (find_project_root() / "outputs" / "feature_panel")
    out = (root / universe_id / f"start={start_month}__end={end_month}" / config_hash)
    # out.mkdir(parents=True, exist_ok=True)
    return out

def save_panel(
    features: pd.DataFrame,
    meta: Dict[str, Any],
    universe_id: str,
    start_month: str,
    end_month: str,
    config_hash: str,
    base_out_dir: Optional[Path] = None
) -> Path:
    """
    Write features.parquet and meta.json to the resolved directory.
    Returns the directory path used.
    """
    out_dir = resolve_features_output_dir(universe_id, start_month, end_month, config_hash, base_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Parquet
    (out_dir / "features.parquet").unlink(missing_ok=True)
    features.to_parquet(out_dir / "features.parquet")
    # Meta
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_dir

def maybe_load_panel(
    universe_id: str,
    start_month: str,
    end_month: str,
    config_hash: str,
    base_out_dir: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """
    If a panel with this signature exists, load and return it; else return None.
    """
    out_dir = resolve_features_output_dir(universe_id, start_month, end_month, config_hash, base_out_dir)
    feats_path = out_dir / "features.parquet"
    if feats_path.exists():
        return pd.read_parquet(feats_path)
    return None


# labels io
def resolve_labels_output_dir(
    universe_id: str,
    start_month: str,  # "YYYY-MM"
    end_month: str,    # "YYYY-MM"
    config_hash: str,
    base_out_dir: Optional[Path] = None
) -> Path:
    """
    Build a deterministic path where the label artifacts live.
    Layout: outputs/labels/<universe_id>/start=YYYY-MM__end=YYYY-MM/<config_hash>/
    """
    root = base_out_dir or (find_project_root() / "outputs" / "labels")
    out = (root / universe_id / f"start={start_month}__end={end_month}" / config_hash)
    # out.mkdir(parents=True, exist_ok=True)
    return out

def save_labels(
    labels: pd.DataFrame,
    meta: Dict[str, Any],
    universe_id: str,
    start_month: str,
    end_month: str,
    config_hash: str,
    base_out_dir: Optional[Path] = None
) -> Path:
    """
    Write labels.parquet and meta.json to the resolved directory.
    Returns the directory path used.
    """
    out_dir = resolve_labels_output_dir(universe_id, start_month, end_month, config_hash, base_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Parquet
    (out_dir / "labels.parquet").unlink(missing_ok=True)
    labels.to_parquet(out_dir / "labels.parquet")
    # Meta
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_dir

def maybe_load_labels(
    universe_id: str,
    start_month: str,
    end_month: str,
    config_hash: str,
    base_out_dir: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """
    If labels with this signature exist, load and return them; else return None.
    """
    out_dir = resolve_labels_output_dir(universe_id, start_month, end_month, config_hash, base_out_dir)
    labels_path = out_dir / "labels.parquet"
    if labels_path.exists():
        return pd.read_parquet(labels_path)
    return None


# backtests io
def resolve_backtests_output_dir(
    params: Dict[str, Any],
    base_out_dir: Optional[Path] = None
) -> Path:
    """
    Build a deterministic path where the backtests artifacts live.
    Layout: outputs/backtests/YYYYMMDD-HHMMSS_<shortsha>
    """
    root = base_out_dir or (find_project_root() / "outputs" / "backtests")

    created_at = utc_now_iso()
    stamp = created_at.replace("-", "").replace(":", "").replace("Z", "").replace("T", "-")
    short_hash = short_sha1_of_json(params, length=8)
    run_id = f"{stamp}_{short_hash}"
    run_dir = root / run_id
    return run_dir

def save_backtest_runs(
    *,
    run_meta: Dict[str, Any],
    params: Dict[str, Any],
    res: Any,  # StrategyResult-like (.name, .equity, .weights, .turnover, .selections, .rebal_dates)
    preds: Optional[pd.DataFrame] = None,
    metrics: Optional[Dict[str, Any]] = None,
    base_out_dir: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Persist core backtest artifacts and the run metadata.

    Files written
    -------------
    - equity.csv            (index=date, single column = strategy name)
    - weights.csv           (index=rebal_date, columns=tickers)        [if present]
    - turnover.csv          (index=rebal_date, col=turnover)           [if present]
    - selections.json       ({YYYY-MM-DD: [tickers, ...]})             [if present]
    - predictions.parquet   (MultiIndex (month,ticker) predictions)     [if provided]
    - metrics.json          (metrics dict)                              [if provided]
    - run_meta.json         (the provided/augmented metadata)

    Parameters
    ----------
    run_meta : dict
        Standardized metadata from `build_run_meta`; this function augments it
        with file paths, run_dir, and run_id before saving to disk.
    params : dict
        Parameters used to construct the folder hash (kept small & stable).
    res : StrategyResult-like
        Result object from the backtester.
    preds : DataFrame, optional
        Predictions to persist (e.g., signals table).
    metrics : dict, optional
        Metrics to persist.
    base_out_dir : Path, optional
        Output root override; defaults to <project_root>/outputs/backtests.

    Returns
    -------
    (run_dir, updated_meta) : (Path, dict)
        Path to created run folder and the metadata dict as saved to disk.
    """
    run_dir = resolve_backtests_output_dir(
        params=params,
        base_out_dir=base_out_dir
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # file paths
    p_equity = run_dir / "equity.csv"
    p_weights = run_dir / "weights.csv"
    p_turnover = run_dir / "turnover.csv"
    p_selections = run_dir / "selections.json"
    p_preds = run_dir / "predictions.parquet" if preds is not None else None
    p_metrics = run_dir / "metrics.json" if metrics is not None else None
    p_meta = run_dir / "run_meta.json"

    # Write core artifacts
    res.equity.to_frame(name=getattr(res, "name", "strategy")).to_csv(p_equity, index=True)
    if getattr(res, "weights", None) is not None:
        res.weights.to_csv(p_weights, index=True)
    if getattr(res, "turnover", None) is not None:
        res.turnover.to_csv(p_turnover, index=True)
    if getattr(res, "selections", None) is not None:
        selections_iso = { to_iso_date(k): v for k, v in res.selections.items() }
        p_selections.write_text(json.dumps(selections_iso, indent=2))
    if preds is not None:
        preds.to_parquet(p_preds, index=True)
    if metrics is not None:
        p_metrics.write_text(json.dumps(metrics, indent=2))

    # update metadata
    root = find_project_root()
    def _rel(p: Optional[Path]) -> Optional[str]:
        return str(p.relative_to(root)) if isinstance(p, Path) and p.exists() else None
    updated_meta = dict(run_meta)  # shallow copy
    updated_meta.update({
        "paths": {
            "run_dir": _rel(run_dir),
            "equity_csv": _rel(p_equity),
            "weights_csv": _rel(p_weights),
            "turnover_csv": _rel(p_turnover),
            "selections_json": _rel(p_selections),
            "predictions_parquet": _rel(p_preds) if p_preds else None,
            "metrics_json": _rel(p_metrics) if p_metrics else None,
            "meta_json": _rel(p_meta),
        }
    })
    updated_meta["run_id"] = run_dir.name
    p_meta.write_text(json.dumps(updated_meta, indent=2))

    return run_dir, updated_meta



# yaml io
def read_yaml(path: Union[str, Path]) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def write_yaml(path: Union[str, Path], data: Any) -> None:
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
