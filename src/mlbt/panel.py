# src/mlbt/panel.py
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

from mlbt.features import vol_1m, mom_P_skip_log_1m
from mlbt.utils import build_panel_meta


def build_feature_panel_v0(
    px_wide: pd.DataFrame,
    month_grid: pd.DataFrame,
    *,
    P: int = 12,
    skip: int = 1,
    min_obs: int = 10,
    annualize: bool = False,
    ddof: int = 0,
    compact_meta: bool = False,
    write: bool = False,
    out_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build Feature Panel v0 (monthly): momentum (P-skip) + within-month volatility.

    This function augments the canonical (month, ticker) grid with:
     - `mom_{P}_{skip}_log_1m`: the momentum over P lookback months, skipping the most recent skip months.
     - `vol_1m` or `ann_vol_1m`: the most recent month's volatility

    Parameters
    ----------
    px_wide : pd.DataFrame
        Wide daily prices (DatetimeIndex, columns=tickers, values=close).
    month_grid : pd.DataFrame
        (month, ticker) grid with at least 'px_M' column.
    P : int, default 12
        Total lookback length in months.
    skip : int, default 1
        Number of most recent months to exclude from the window (e.g. skip=1
        means end the lookback at M-1).
    min_obs : int, default 10
        Minimum daily-return observations required for vol_1m.
    annualize : bool, default False
        Annualize vol_1m via sqrt(252) and name 'ann_vol_1m'.
    ddof : int, default 0
        Degrees of freedom for stdev (0=population, 1=sample).
    compact_meta : bool, default False
        Switches of certain information that is repeated when similar meta is created from several functions within a pipeline.
    write : bool, default False
        If True, write features.parquet + meta.json to disk.
    out_dir : pathlib.Path | None
        Optional base output directory override (useful for tests).

    Returns
    -------
    (features, meta) : Tuple[pd.DataFrame, dict]
        `features` includes the original grid columns plus:
         - `mom_{P}_{skip}_log_1m`
         - `vol_1m` or `ann_vol_1m`
        `meta` is a standardized dict with coverage, universe summary, paramerization, and a config hash.
    """
    # Start from the canonical grid
    features = month_grid.copy()

    # Momentum feature (P-skip) as log return between px_{M-skip} and px_{M-P}
    features = mom_P_skip_log_1m(
        features=features,
        P=P,
        skip=skip
    )
    mom_col = f"mom_{P}_{skip}_log_1m"

    # Volatility feature within month
    features = vol_1m(
        px_wide=px_wide,
        features=features,
        min_obs=min_obs,
        annualize=annualize,
        ddof=ddof
    )
    vol_col = "ann_vol_1m" if annualize else "vol_1m"

    # Build metadata
    params = {
        "P": P,
        "skip": skip,
        "min_obs": min_obs,
        "annualize": annualize,
        "ddof": ddof,
    }
    feature_list = [mom_col, vol_col]

    meta = build_panel_meta(
        month_grid=month_grid,
        feature_names=feature_list,
        params=params,
        panel_name="feature_panel",
        panel_version="v0",
        compact_meta=compact_meta
    )

    if write:
        from mlbt.io import save_panel
        # to save we need non_compact meta
        _meta = build_panel_meta(
            month_grid=month_grid,
            feature_names=feature_list,
            params=params,
            panel_name="feature_panel",
            panel_version="v0",
            compact_meta=False
        )
        save_panel(
            features=features,
            meta=_meta,
            base_out_dir=out_dir,
        )

    return features, meta
