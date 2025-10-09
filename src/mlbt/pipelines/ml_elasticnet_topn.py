# src/mlbt/pipelines/ml_elasticnet_topn.py
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

from mlbt.strategy_result import StrategyResult
from mlbt.utils import build_run_meta
from mlbt.utils import validate_month_grid_index, validate_px_wide_index
from mlbt.trainer import walkforward_elasticnet_v0
from mlbt.backtest_engines import backtest_topn
from mlbt.io import save_backtest_runs


def run_elasticnet_topn_v0(
    px_wide: pd.DataFrame,
    month_grid: pd.DataFrame,
    *,
    feature_params: Optional[Dict] = None,
    label_params: Optional[Dict] = None,
    model_params: Optional[Dict] = None,
    backtest_params: Optional[Dict] = None,
    run_name: Optional[str] = None,
    save: bool = False,
    out_dir: Optional[Path] = None
) -> Tuple["StrategyResult", Dict]:
    """
    Orchestrate a Top-N backtest from walkforward_elasticnet_v0.

    Builds feature and label panels from `px_wide`/`month_grid`, performs
    walk-forward ElasticNet training/prediction, validates the resulting
    predictions, executes a Top-N backtest, computes metrics, and persists
    artifacts for reproducible analysis.

    Parameters
    ----------
    px_wide : pandas.DataFrame
        Daily prices (DatetimeIndex, columns = tickers).

    month_grid : pandas.DataFrame
        MultiIndex (month, ticker) universe grid controlling panel alignment.

    feature_params : dict, optional
        Parameters forwarded to your feature builder (e.g. for v0: {"P": 12, "skip": 1, "min_obs": 10, "annualize": False, "ddof": 0}). If omitted keys exist, sensible defaults from the builder are used.

    label_params : dict, optional
        Parameters for the label builder.

    model_params : dict, optional
        ElasticNet hyperparameters and training knobs, e.g. for v0: {"alpha": 0.001, "l1_ratio": 0.5, "random_state": 0, "min_train_samples": 100}.

    backtest_params : dict, optional
        Parameters for the backtesting engine, e.g. {"rank_col": y_pred", "N": 10, "cost_bps": 4.0, "strict": True}.

    run_name : str, optional
        Friendly label appearing in saved metadata and filenames.

    save : bool, default=False
        If True, persist predictions, backtest outputs, metrics, and metadata.

    out_dir : Path, optional
        Root directory for the run folder.

    Returns
    -------
    (res, meta) : Tuple[StrategyResult, dict]
        `res`: StrategyResult of the Top-N backtest.
        `meta`: metadata combining panel/model params, universe info, and paths
        to saved artifacts.
    """
    # some guards
    validate_px_wide_index(px_wide)
    validate_month_grid_index(month_grid)

    # create predictions
    feature_params = {} if feature_params is None else feature_params
    label_params = {} if label_params is None else label_params
    model_params = {} if model_params is None else model_params
    preds, pred_meta = walkforward_elasticnet_v0(
        px_wide=px_wide,
        month_grid=month_grid,
        **feature_params,
        **label_params,
        **model_params,
        compact_meta=True,
        out_dir=out_dir
    )

    # run top-n backtest
    backtest_params = {} if backtest_params is None else backtest_params
    res, backtest_params = backtest_topn(
        px_wide=px_wide,
        predictions=preds,
        **backtest_params,
        name=run_name
    )

    # compute metrics
    try:
        m = res.compute_metrics().to_dict()
    except Exception:
        m = None

    # meta data
    run_meta = build_run_meta(
        predictions=preds,
        res=res,
        run_name=run_name,
        backtest_params=backtest_params,
        runner_name="elasticnet_topn",
        runner_version="v0",
        predictions_meta=pred_meta,
        metrics=m if isinstance(m, dict) else None
    )

    if save:
        run_dir, saved_meta = save_backtest_runs(
            run_meta=run_meta,
            hashing_params=run_meta["backtest_params"],
            res=res,
            preds=preds,
            metrics=m if isinstance(m, dict) else None,
            base_out_dir=out_dir,
        )
        run_meta = saved_meta

    return res, run_meta