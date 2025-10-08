# src/mlbt/trainer.py
"""
Training utilities for walk-forward model fitting and prediction.

This module contains functions that:
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

from mlbt.utils import build_trainer_meta
from mlbt.utils import validate_columns_exist



# ---------------- Convenience ----------------

def walkforward_elasticnet_v0(
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
    compact_meta: bool = False,
    write: bool = False,
    out_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Walk-forward ElasticNet using v0 feature/label builders (convenience wrapper).

    Builds panels via `build_feature_panel_v0` and `build_label_panel_v0`, then
    delegates to `walkforward_elasticnet`.

    Parameters
    ----------
    px_wide : pandas.DataFrame
        Daily prices in wide format (DatetimeIndex, one column per ticker). Used by panel builders to derive monthly features/labels.
    month_grid : pandas.DataFrame
        MultiIndex (month, ticker) grid defining the evaluation universe and months. Panel builders use this grid to align outputs.
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
    compact_meta : bool, default False
        Switches of certain information that is repeated when similar meta is created from several functions within a pipeline.
    write : bool, default=False
        If True, panel builders may persist intermediate feature/label outputs
        (implementation-dependent).
    out_dir : pathlib.Path, optional
        Target directory for any optional writes by panel builders.
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
        compact_meta=True,
        write=write,
        out_dir=out_dir
    )    
    labels, label_meta = build_label_panel_v0(
        month_grid=month_grid,
        compact_meta=True,
        write=write,
        out_dir=out_dir
    )
    feature_cols = feature_meta["features"]
    label_col = "y_ret_1m"

    # now call core function
    preds, meta = walkforward_elasticnet(
        features=features,
        labels=labels,
        feature_cols=feature_cols,
        label_col=label_col,
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=random_state,
        min_train_samples=min_train_samples,
        features_meta=feature_meta,
        label_meta=label_meta,
        compact_meta=compact_meta
    )

    return preds, meta



# ---------------- Core ----------------

def walkforward_elasticnet(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    feature_cols: list[str],
    label_col: str,
    alpha: float = 0.001,
    l1_ratio: float = 0.5,
    random_state: int = 0,
    min_train_samples: int = 100,
    features_meta: Optional[Dict] = None,
    label_meta: Optional[Dict] = None,
    compact_meta: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    """
    Walk-forward train/predict with ElasticNet given prebuilt panels.

    Parameters
    ----------
    features : pd.DataFrame
        MultiIndex (month, ticker) features panel. May contain extra columns.
    labels : pd.DataFrame
        MultiIndex (month, ticker) labels panel. Must contain `label_col`.
    feature_cols : list of str
        Names of feature columns to feed into the model.
    label_col : str
        Name of the target column in `labels`.
    alpha : float, default=0.001
        ElasticNet `alpha` regularization strength.
    l1_ratio : float, default=0.5
        ElasticNet `l1_ratio` (0 = ridge, 1 = lasso).
    random_state : int, default=0
        Seed for ElasticNet to ensure deterministic fits.
    min_train_samples : int, default=100
        Minimum number of training rows required before making any prediction
        for a given month. Months with fewer rows are skipped.
    features_meta, label_meta : dict, optional
        Optional meta dicts to embed into the returned metadata for traceability.
    compact_meta : bool, default False
        Switches of certain information that is repeated when similar meta is created from several functions within a pipeline.

    Returns
    -------
    (preds, meta) : (pd.DataFrame, dict)
        preds has columns ["y_true","y_pred"], index=(month,ticker), sorted.
        meta contains model params, coverage, universe summary, selected features, training diagnostics, and last-month coefficients (if available).
    """
    # join panels
    panel = features.join(labels.drop(columns=labels.columns.intersection(features.columns)), how="left")
    panel = panel.dropna(how="any").sort_index()

    # some guards
    validate_columns_exist(panel, feature_cols)
    
    months = panel.index.get_level_values("month").unique().sort_values()

    all_preds = []
    train_rows_by_month: Dict[str, int] = {}
    last_fitted_en = None

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

    coef_last_month: Dict[str, float] = {}
    if len(all_preds) == 0:
        empty_idx = features.index[:0]
        preds =  pd.DataFrame(index=empty_idx, columns=["y_true", "y_pred"], dtype=float)
    else:
        preds = pd.concat(all_preds).sort_index()
        
        if last_fitted_en is not None:
            coef_arr = getattr(last_fitted_en, "coef_", None)
            if coef_arr is not None and len(coef_arr) == len(feature_cols):
                coef_last_month = dict(zip(feature_cols, coef_arr.tolist()))

    # meta data
    model_params = {
        "type": "ElasticNet",
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "random_state": random_state
    }
    training_meta = {
        "min_train_samples": min_train_samples,
        "months_evaluated": preds.index.get_level_values("month").nunique() if not preds.empty else 0,
        "train_rows_by_month": train_rows_by_month
    }
    meta = build_trainer_meta(
        predictions=preds,
        model_params=model_params,
        training_meta=training_meta,
        features_used=feature_cols,
        features_meta=features_meta,
        label_meta=label_meta,
        coef_last_month=coef_last_month,
        compact_meta=compact_meta
    )

    return preds, meta
