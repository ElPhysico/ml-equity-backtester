# src/mlbt/utils.py
import pandas as pd
import hashlib
from pathlib import Path
from datetime import datetime, UTC
import os
import json
import inspect
import logging
from collections.abc import Sequence, Callable, Mapping, MutableMapping
from typing import get_origin

PROJECT_SENTINELS = ("pyproject.toml", ".git")


# ---------------- Validators ----------------

def validate_px_wide_index(px_wide: pd.DataFrame) -> None:
    """
    Raise ValueError if `px_wide` index is not a unique DatetimeIndex.
    """
    if not isinstance(px_wide.index, pd.DatetimeIndex):
        raise ValueError("Index is not DatetimeIndex.")
    if not px_wide.index.is_unique:
        raise ValueError("Duplicate timestamps in index.")
    
def validate_px_wide_range(
    px_wide: pd.DataFrame,
    data_start: str | pd.Timestamp,
    data_end: str | pd.Timestamp
) -> None:
    """
    Raises ValueError if px_wide is empty or does not cover the time period indicated by data_start to data_end.
    """
    if px_wide.empty:
        raise ValueError("Daily prices table is empty.")
    if px_wide.index.min() > pd.Timestamp(data_start):
        raise ValueError(f"data_start ({data_start}) is earlier than earliest record in daily prices table ({px_wide.index.min()})")
    if px_wide.index.max() < pd.Timestamp(data_end):
        raise ValueError(f"data_end ({data_end}) is later than latest record in daily prices table ({px_wide.index.max()})")

def validate_month_grid_index(month_grid: pd.DataFrame) -> None:
    """
    Raise ValueError if `month_grid` does not use a MultiIndex
    with levels named ('month', 'ticker').
    """
    if not isinstance(month_grid.index, pd.MultiIndex):
        raise ValueError("Month/ticker grids must be indexed by MultiIndex (month, ticker).")
    if "month" not in month_grid.index.names or "ticker" not in month_grid.index.names:
        raise ValueError("Month/ticker grids index must have levels ['month','ticker'].")
    
def validate_columns_exist(
    df: pd.DataFrame,
    cols: str | Sequence[str]
) -> None:
    """
    Raise ValueError if object of the required column(s) are missing in `df`. Accepts a single column name or a sequence of names.
    """
    required = [cols] if isinstance(cols, str) else list(cols)
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required column(s): {missing}")
    
def validate_config(
    config: dict[str, object],
    required: str | Sequence[str],
    log_name: str | None = None
) -> None:
    """
    Raise ValueError if object of the required keyword(s) are missing in config. Accepts a single keyword or a sequence of keywords.
    """
    required = [required] if isinstance(required, str) else list(required)
    missing = [c for c in required if c not in config]
    
    if missing:
        raise ValueError(f"Missing required keywords in {'config' if log_name is None else log_name}: {missing}")

# ---------------- Helper ----------------

def _is_sequence_but_not_str(x: object) -> bool:
    return isinstance(x, Sequence) and not isinstance(x, (str, bytes))

def normalize_to_list(
    x: object,
    *,
    element_type: type | tuple[type, ...] | None = None,
    allow_none: bool = True
) -> list | None:
    """
    Normalize a value to a list:
      - None -> None (if allow_none) else [None]
      - scalar -> [scalar]
      - sequence (not str/bytes) -> list(sequence)

    Optionally validates each element's type via `element_type`.
    """
    if x is None:
        return None if allow_none else [None]

    if _is_sequence_but_not_str(x):
        out = list(x)
    else:
        out = [x]

    if element_type is not None:
        bad = [i for i, v in enumerate(out) if v is not None and not isinstance(v, element_type)]
        if bad:
            raise TypeError(
                f"Elements at positions {bad} are not of type {element_type}."
            )
    return out

def broadcast_lists(
    *lists: list | None,
    target_len: int | None = None
) -> tuple[list, ...]:
    """
    Broadcast multiple (possibly None) lists to a common length.
    - object list of length 1 is repeated to target_len.
    - lists already at target_len are kept.
    - None becomes [None] * target_len.
    - If there are conflicting lengths (>1 and unequal), raise ValueError.
    """
    # Determine target length if not given
    lengths = [len(x) for x in lists if isinstance(x, list)]
    if target_len is None:
        target_len = max(lengths) if lengths else 1

    # Validate non-broadcastable lengths
    fixed_lengths = {L for L in lengths if L > 1}
    if fixed_lengths and (len(fixed_lengths) > 1) and (target_len not in fixed_lengths):
        raise ValueError(f"Conflicting lengths: {sorted(fixed_lengths)} (target_len={target_len}).")

    out = []
    for x in lists:
        if x is None:
            out.append([None] * target_len)
        elif len(x) == 1:
            out.append(x * target_len)
        elif len(x) == target_len:
            out.append(x)
        else:
            # len(x) > 1 and != target_len
            raise ValueError(f"Cannot broadcast length {len(x)} to {target_len}.")
    return tuple(out)


def bind_config(
        func: Callable[..., object],
        config: dict[str, object]
) -> dict[str, object]:
    """
    Forward only keys that the func accepts.
    """
    sig = inspect.signature(func)
    return {
        k: v for k, v in config.items()
        if k in sig.parameters
    }

def _is_dict_annotation(ann: object) -> bool:
    """Return True if the annotation is dict-like (dict/dict/Mapping/MutableMapping)."""
    if ann is inspect._empty:
        return False
    origin = get_origin(ann) or ann
    return origin in (dict, dict, Mapping, MutableMapping)

def dict_param_names(func: Callable[..., object]) -> set[str]:
    """Names of parameters annotated as dict-like."""
    sig = inspect.signature(func)
    return {
        name
        for name, p in sig.parameters.items()
        if _is_dict_annotation(p.annotation)
    }

def bind_nested_configs(
    func: Callable[..., object],
    cfg: Mapping[str, object],
    schema_map: Mapping[str, Callable[..., object]],
) -> dict[str, object]:
    """
    Re-bind nested dict params using schema_map {param_name: sub_func}.
    Only processes keys present in cfg and whose values are dicts.
    If the func's parameter isn't annotated as dict-like, it will still be processed
    if it's explicitly listed in schema_map (opt-in override).
    """
    out = dict(bind_config(func, cfg))  # start with top-level binding

    dict_like = dict_param_names(func)
    for pname, sub_func in schema_map.items():
        if pname in out:
            val = out[pname]
            if isinstance(val, dict) and (pname in dict_like or pname in schema_map):
                out[pname] = bind_config(sub_func, val)
    return out

def clean_dict(
    dic: dict[str, object],
    allowed: str | Sequence[str]
) -> dict[str, object]:
    if dic is not None:
        invalid = set(dic) - set(allowed)
        if invalid:
            logging.warning(
                "Ignoring unsupported keys: %s",
                ", ".join(sorted(invalid))
            )
        dic = {k: v for k, v in dic.items() if k in allowed}
    return dic

def sha1_of_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def short_sha1_of_json(obj: dict, length: int = 8) -> str:
    """Deterministic short hash for run IDs (stable across Python processes)."""
    return sha1_of_str(json.dumps(obj, sort_keys=True, separators=(",", ":")))[:length]

def utc_now_iso() -> str:
    # ISO 8601 with 'Z' suffix
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

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

def coverage_from_grid(month_grid: pd.DataFrame) -> tuple[str, str]:
    months = month_grid.index.get_level_values("month")
    start = str(months.min())  # "YYYY-MM"
    end   = str(months.max())  # "YYYY-MM"
    return start, end

def config_hash(universe_hash: str, params: dict[str, object], feature_names: list[str]) -> str:
    # Stable, readable-ish
    key = {
        "universe_hash": universe_hash,
        "params": params,
        "features": feature_names,
    }
    return sha1_of_str(json.dumps(key, sort_keys=True))

def tickers_from_grid(month_grid: pd.DataFrame) -> list[str]:
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


def universe_summary_from_grid(month_grid: pd.DataFrame) -> dict[str, object]:
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


# ---------------- Meta Builders ----------------

def build_panel_meta(
    month_grid: pd.DataFrame,
    *,
    feature_names: list[str],
    params: dict[str, object],
    panel_name: str,
    panel_version: str = "v0",
    extra: dict[str, object] | None = None,
    compact_meta: bool = False
) -> dict[str, object]:
    """
    Construct a standardized metadata dict for object monthly panel.

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
    compact_meta : bool, default False
        Switches of certain information that is repeated when similar meta is created from several functions within a pipeline.

    Returns
    -------
    meta : dict
        Standardized metadata including created_at, coverage, universe,
        panel identifier, features, params, and a config_hash.
    """
    start_month, end_month = coverage_from_grid(month_grid)
    univ = universe_summary_from_grid(month_grid)
    cfg_hash = config_hash(univ["hash"], params, feature_names)

    meta: dict[str, object] = {
        "created_at": utc_now_iso(),
        "panel": {"name": panel_name, "version": panel_version}
    }

    if not compact_meta:
        meta["data_coverage"] = {"start": start_month, "end": end_month}
        meta["universe"] = univ

    meta.update({
        "features": feature_names,
        "params": params,
        "config_hash": cfg_hash,
    })
    if extra:
        # Shallow merge; caller controls collisions intentionally.
        meta.update(extra)
    return meta


def build_trainer_meta(
    predictions: pd.DataFrame,
    *,
    model_params: dict[str, object],
    training_meta: dict[str, object],
    features_used: list[str] | None = None,
    features_meta: dict[str, object] | None = None,
    label_meta: dict[str, object] | None = None,
    coef_last_month: dict[str, float] | None = None,
    extra: dict[str, object] | None = None,
    compact_meta: bool = False
) -> dict[str, object]:
    """
    Construct a standardized metadata dictionary for object trainer.

    Parameters
    ----------
    predictions : pandas.DataFrame
        MultiIndex (month, ticker) table used to infer coverage and universe summary.
    model_params : dict
        Parameters used for the training model.
    training_meta : dict
        dictionary containing training information, such as min_train_samples, number of months that have been evaluated, training data per month, etc.
    features_used : list of str, optional
        list of strings containing the names of the features used in training.
    features_meta : dict, optional
        dictionary containing additional information about the features as returned by build_feature_panel_v0 for instance.
    label_meta : dict, optional
        dictionary containing additional information about the label as returned by build_label_panel_v0 for instance.
    coef_last_month : dict, optional
        dictionary containing the coefficients per feature for the last month.    
    extra : dict, optional
        Free-form additional fields merged at top level.
    compact_meta : bool, default False
        Switches of certain information that is repeated when similar meta is created from several functions within a pipeline.

    Returns
    -------
    meta : dict
        Standardized trainer metadata including created_at, coverage, universe, model parameters, training meta, features meta, label meta, last month's coefficients, and object optional extra information.
    """
    start_month, end_month = coverage_from_grid(predictions)
    univ = universe_summary_from_grid(predictions)

    meta: dict[str, object] = {
        "created_at": utc_now_iso()
    }

    if not compact_meta:
        meta["data_coverage"] = {"start": start_month, "end": end_month}
        meta["universe"] = univ
    
    meta.update({
        "model": model_params,
        "training": training_meta
    })
    meta["features_used"] = [] if features_used is None else features_used
    if features_meta:
        meta["features_meta"] = features_meta
    if label_meta:
        meta["label_meta"] = label_meta
    if coef_last_month:
        meta["coef_last_month"] = coef_last_month
    if extra:
        meta["extra"] = extra

    return meta


def build_run_meta(
    predictions: pd.DataFrame,
    res: object,
    *,
    name: str | None,
    backtest_params: dict[str, object],
    runner_name: str,
    runner_version: str | None = None,
    predictions_meta: dict | None = None,
    metrics: dict[str, object] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """
    Construct a standardized metadata dictionary for object backtest or experiment run.

    Parameters
    ----------
    predictions : pandas.DataFrame
        MultiIndex (month, ticker) table used to infer coverage and universe summary.
    res : object
        StrategyResult-like object with `.rebal_dates` and `.name`; adds brief strategy info.
    name : str, optional
        Friendly label chosen by the user (appears in logs and filenames).
    backtest_params : dict
        Parameters governing the backtest (e.g., rank_col, N, cost_bps, strict, model_cfg, etc.).
    runner_name : str
        Name of the orchestration function (e.g., "topn_from_predictions").
    runner_version : str, optional
        Implementation version tag.
    predictions_meta : dict, optional
        Optional dictionary containing meta information for the prediction generation.
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
            "strategy_start": to_iso_date(res.start_date),
            "strategy_end": to_iso_date(res.end_date),
            "n_rebalances": res.n_rebalances,
            "first_rebalance": to_iso_date(res.rebal_dates[0]) if getattr(res, "rebal_dates", None) is not None and len(res.rebal_dates) else None,
            "last_rebalance": to_iso_date(res.rebal_dates[-1]) if getattr(res, "rebal_dates", None) is not None and len(res.rebal_dates) else None,
        }
    except Exception:
        strat_info = None

    meta: dict[str, object] = {
        "created_at": utc_now_iso(),
        "runner": {"name": runner_name, "version": "-" if runner_version is None else runner_version},
        "name": name,
        "data_coverage": {"start": start_month, "end": end_month},
        "universe": univ,
        "backtest_params": backtest_params,
    }
    if strat_info:
        meta["strategy"] = strat_info
    if predictions_meta:
        meta["predictions_meta"] = predictions_meta
    if metrics:
        meta["metrics"] = metrics
    if extra:
        meta.update(extra)

    return meta




