# src/mlbt/trainer.py
"""
Training utilities for walk-forward model fitting and prediction.

This module contains functions that:
- Build feature and label panels from prices and a monthly grid.
- Train models in a strictly forward-looking manner (walk-forward).
- Return tidy predictions indexed by (month, ticker) together with rich metadata
  useful for reproducibility (coverage window, universe, hyperparameters,
  features used, last-fit coefficients, and simple training diagnostics).

The functions here do not execute backtests; they are intentionally decoupled
from portfolio construction. Use `mlbt.backtest_topn` or a pipeline in
`mlbt.pipelines.*` to transform predictions into strategy performance.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet

from mlbt.panel import build_feature_panel_v0
from mlbt.labels import build_label_panel_v0

from mlbt.utils import sha1_of_str, utc_now_iso, universe_id_from_grid, coverage_from_grid


def walkforward_predict_v0(
    px_wide: pd.DataFrame,
    month_grid: pd.DataFrame,
    *,
    alpha: float = 0.001,
    l1_ratio: float = 0.5,
    random_state: int = 0,
    min_train_samples: int = 100,
    P: int = 12,
    skip: int = 1,
    min_obs: int = 10,
    annualize: bool = False,
    ddof: int = 0,
    write: bool = False,
    out_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Walk-forward train/predict with ElasticNet on monthly panels.

    Builds features and labels for each (month, ticker), then walks forward
    month by month: using only data strictly prior to month `m` to fit a
    pipeline (StandardScaler → ElasticNet), and predicting on month `m`.
    Returns a tidy predictions DataFrame and a metadata dict capturing model,
    data coverage, and simple diagnostics.

    Parameters
    ----------
    px_wide : pandas.DataFrame
        Daily prices in wide format (DatetimeIndex, one column per ticker).
        Used by panel builders to derive monthly features/labels.

    month_grid : pandas.DataFrame
        MultiIndex (month, ticker) grid defining the evaluation universe and
        months. Panel builders use this grid to align outputs.

    alpha : float, default=0.001
        ElasticNet `alpha` regularization strength.

    l1_ratio : float, default=0.5
        ElasticNet `l1_ratio` (0 = ridge, 1 = lasso).

    random_state : int, default=0
        Seed for ElasticNet to ensure deterministic fits.

    min_train_samples : int, default=100
        Minimum number of training rows required before making any prediction
        for a given month. Months with fewer rows are skipped.

    P : int, default=12
        Lookback length used by `build_feature_panel_v0` (e.g., momentum window).

    skip : int, default=1
        Skip parameter for features (e.g., “12-1” momentum uses skip=1).

    min_obs : int, default=10
        Minimum observed data points required for a ticker to contribute to
        feature computation.

    annualize : bool, default=False
        Whether certain feature computations should be annualized (delegated to
        the feature builder).

    ddof : int, default=0
        Delta degrees of freedom for variance-related features.

    write : bool, default=False
        If True, panel builders may persist intermediate feature/label outputs
        (implementation-dependent).

    out_dir : pathlib.Path, optional
        Target directory for any optional writes by panel builders.

    Returns
    -------
    (preds, meta) : Tuple[pandas.DataFrame, dict]
        `preds`: DataFrame indexed by (month, ticker) with columns:
            - "y_true": realized 1m forward return
            - "y_pred": model prediction for the same target
        `meta`: dict with keys:
            - "created_at": ISO timestamp
            - "data_coverage": {"start", "end"} month strings
            - "universe": {"tickers", "count", "hash", "id"}
            - "model": {"type", "alpha", "l1_ratio", "random_state"}
            - "features_used": list of feature column names
            - "features_params": {"P","skip","min_obs","annualize","ddof"}
            - "training": {"min_train_samples","months_evaluated",
                           "train_rows_by_month"}
            - "coef_last_month": {feature: coefficient} for the last fitted month
              (empty dict if nothing was fitted)

    Notes
    -----
    - Strictly forward-looking: the model for month `m` is fitted only on
      months `< m`.
    - The function skips early months until `min_train_samples` is met.
    - No backtesting or portfolio construction occurs here; downstream code
      should pass `preds` into a selector/execution engine (e.g., Top-N).
    """

    # obtain features and labels
    features, feature_meta = build_feature_panel_v0(
        px_wide=px_wide,
        month_grid=month_grid,
        P=P,
        skip=skip,
        min_obs=min_obs,
        annualize=annualize,
        ddof=ddof,
        write=write,
        out_dir=out_dir
    )
    
    labels, _ = build_label_panel_v0(
        month_grid=month_grid,
        write=write,
        out_dir=out_dir
    )
    feature_cols = feature_meta["features"]
    label_col = "y_ret_1m"

    panel = features.join(labels.drop(columns=labels.columns.intersection(features.columns)), how="left")
    panel = panel.dropna(how="any").sort_index()

    all_preds = []
    train_rows_by_month: dict[str, int] = {}
    months = panel.index.get_level_values("month").unique().sort_values()

    # walk forward training
    for m in months:
        train_set = panel[panel.index.get_level_values("month") < m]
        if len(train_set) < min_train_samples:
            continue
        test_set = panel[panel.index.get_level_values("month") == m]        

        X_train = train_set[feature_cols].to_numpy()
        y_train = train_set[label_col].to_numpy()
        X_test = test_set[feature_cols].to_numpy()
        y_test = test_set[label_col].to_numpy()

        model = make_pipeline(
            StandardScaler(),
            ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
        )
        model.fit(X_train, y_train)
        train_rows_by_month[str(m)] = len(X_train)
        last_fitted_en = model.named_steps["elasticnet"]
        
        y_pred = model.predict(X_test)
        preds = pd.DataFrame(
            {"y_true": y_test, "y_pred": y_pred},
            index=test_set.index
        ).sort_index()
        all_preds.append(preds)

    if len(all_preds) == 0:
        empty_idx = features.index[:0]
        preds =  pd.DataFrame(index=empty_idx, columns=["y_true", "y_pred"], dtype=float)
        coef_last_month = {}
    else:
        preds = pd.concat(all_preds).sort_index()
        coef_last_month = {}
        if last_fitted_en is not None:
            coef_arr = getattr(last_fitted_en, "coef_", None)
            if coef_arr is not None and len(coef_arr) == len(feature_cols):
                coef_last_month = dict(zip(feature_cols, coef_arr.tolist()))

    # metadata
    start_month, end_month = coverage_from_grid(month_grid)
    universe_id = universe_id_from_grid(month_grid)
    tickers = sorted(set(month_grid.index.get_level_values("ticker")))
    universe_hash = sha1_of_str(",".join(tickers))

    meta = {
        "created_at": utc_now_iso(),
        "data_coverage": {"start": start_month, "end": end_month},
        "universe": {
            "tickers": tickers,
            "count": len(tickers),
            "hash": universe_hash,
            "id": universe_id
        },
        "model": {
            "type": "ElasticNet",
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "random_state": random_state
        },
        "features_used": feature_cols,
        "features_params": {
            "P": P,
            "skip": skip,
            "min_obs": min_obs,
            "annualize": annualize,
            "ddof": ddof
        },
        "training": {
            "min_train_samples": min_train_samples,
            "months_evaluated": preds.index.get_level_values("month").nunique() if not preds.empty else 0,
            "train_rows_by_month": train_rows_by_month
        },
        "coef_last_month": coef_last_month
    }
    
    return preds, meta