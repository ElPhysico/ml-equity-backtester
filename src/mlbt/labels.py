# src/mlbt/labels.py
"""
Label engineering utilities.

This module defines how forward-looking labels are constructed from the
month-end trading grid. Labels represent outcomes that are *unknown at
rebalance time* and are therefore valid targets for model training.

Provided functions
------------------
- ret_1m : append px_Mplus1 and y_ret_1m (next-month price and simple return).
- build_label_panel_v0 : orchestrate label creation, metadata assembly,
  and optional saving via mlbt.io.save_labels.

Naming conventions
------------------
- Columns ending in '_1m' refer to one-month forward quantities.
- Any quantity from month t+1 relative to month t (e.g. px_Mplus1,
  y_ret_1m) is considered a *label*.
- Index is always (month: Period['M'], ticker: str) to align with
  the feature panels and backtesting engine.
"""
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from mlbt.utils import sha1_of_str, utc_now_iso, universe_id_from_grid, coverage_from_grid, config_hash


def build_label_panel_v0(
        month_grid: pd.DataFrame,
        *,
        write: bool = False,
        out_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Docstring to be written.
    """
    labels = month_grid.copy()
    
    # next (t+1) month price and return t -> t+1
    labels = ret_1m(labels=labels)
    px_Mplus1_col = "px_Mplus1"
    y_ret_1m_col = "y_ret_1m"

    # metadata
    start_month, end_month = coverage_from_grid(month_grid)
    universe_id = universe_id_from_grid(month_grid)
    tickers = sorted(set(month_grid.index.get_level_values("ticker")))
    universe_hash = sha1_of_str(",".join(tickers))
    params = {

    }
    labels_list = [px_Mplus1_col, y_ret_1m_col]

    meta = {
        "created_at": utc_now_iso(),
        "data_coverage": {"start": start_month, "end": end_month},
        "universe": {
            "tickers": tickers,
            "count": len(tickers),
            "hash": universe_hash,
            "id": universe_id
        },
        "labels": labels_list,
        "params": params
    }
    cfg_hash = config_hash(universe_hash, params, labels_list)
    meta["config_hash"] = cfg_hash

    if write:
        from mlbt.io import save_labels
        save_labels(
            labels=labels,
            meta=meta,
            universe_id=universe_id,
            start_month=start_month,
            end_month=end_month,
            config_hash=cfg_hash,
            base_out_dir=out_dir
        )

    return labels, meta


def ret_1m(
    labels: pd.DataFrame
) -> pd.DataFrame:
    """
    Docstring to be written.
    """
    gb = labels.groupby("ticker", observed=True)
    px_Mplus1 = gb["px_M"].shift(-1)
    y_ret_1m = gb["px_M"].transform(lambda s: s.shift(-1) / s)

    labels = labels.join([px_Mplus1.to_frame("px_Mplus1"), y_ret_1m.to_frame("y_ret_1m")], how="left")

    return labels