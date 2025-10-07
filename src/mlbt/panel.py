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
    write: bool = False,
    out_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build Feature Panel v0 (monthly): momentum (P-skip) + within-month volatility.

    Parameters
    ----------
    px_wide : pd.DataFrame
        Wide daily prices (DatetimeIndex, columns=tickers, values=close).
    month_grid : pd.DataFrame
        (month, ticker) grid with at least 'px_M' column.
    P, skip : int
        Momentum window (e.g., 12-1).
    min_obs : int
        Minimum daily-return observations required for vol_1m.
    annualize : bool
        Annualize vol_1m via sqrt(252) and name 'ann_vol_1m'.
    ddof : int
        Degrees of freedom for stdev (0=population, 1=sample).
    write : bool
        If True, write features.parquet + meta.json to disk.
    out_dir : pathlib.Path | None
        Optional base output directory override (useful for tests).

    Returns
    -------
    (features_df, meta_dict)
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
    )

    if write:
        from mlbt.io import save_panel
        save_panel(
            features=features,
            meta=meta,
            universe_id=meta["universe"]["id"],
            start_month=meta["data_coverage"]["start"],
            end_month=meta["data_coverage"]["end"],
            config_hash=meta["config_hash"],
            base_out_dir=out_dir,
        )

    return features, meta
