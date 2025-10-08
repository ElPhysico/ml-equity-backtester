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

from mlbt.utils import build_panel_meta


def build_label_panel_v0(
    month_grid: pd.DataFrame,
    *,
    compact_meta: bool = False,
    write: bool = False,
    out_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build Label Panel v0 (monthly): next-month price and 1M forward return.

    This function augments the canonical (month, ticker) grid with:
      - `px_Mplus1`: the next month’s end-of-month price
      - `y_ret_1m` : the 1-month forward return from month-end to next month-end

    Parameters
    ----------
    month_grid : pd.DataFrame
        MultiIndex (month, ticker) grid used as the base alignment for labels.
        Must include the columns required by `ret_1m(...)` to compute forward
        price/return.
    compact_meta : bool, default False
        Switches of certain information that is repeated when similar meta is created from several functions within a pipeline.
    write : bool, default=False
        If True, persist the labels and metadata via `mlbt.io.save_labels(...)`.
    out_dir : pathlib.Path, optional
        Optional base output directory override (useful for tests).

    Returns
    -------
    (labels, meta) : Tuple[pd.DataFrame, dict]
        `labels` includes the original grid columns plus:
          - "px_Mplus1"
          - "y_ret_1m"
        `meta` is a standardized dict with coverage, universe summary,
        parameterization, and a config hash.
    """
    labels = month_grid.copy()
    
    # next (t+1) month price and return t -> t+1
    labels = ret_1m(labels=labels)
    px_Mplus1_col = "px_Mplus1"
    y_ret_1m_col = "y_ret_1m"

    # metadata

    params = {

    }
    labels_list = [px_Mplus1_col, y_ret_1m_col]

    meta = build_panel_meta(
        month_grid=month_grid,
        feature_names=labels_list,
        params=params,
        panel_name="label_panel",
        panel_version="v0",
        compact_meta=compact_meta
    )

    if write:
        from mlbt.io import save_labels
        # to save we need non_compact meta
        _meta = build_panel_meta(
            month_grid=month_grid,
            feature_names=labels_list,
            params=params,
            panel_name="label_panel",
            panel_version="v0",
            compact_meta=False
        )
        save_labels(
            labels=labels,
            meta=_meta,
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