# src/mlbt/pipelines/topn_runner.py
import pandas as pd
from pathlib import Path

from mlbt.specs.strategy_result import StrategyResult
from mlbt.backtest_engines.backtest_topn import backtest_topn
from mlbt.utils import build_run_meta
from mlbt.utils import validate_month_grid_index
from mlbt.io import save_backtest_runs


def run_topn_from_predictions(
    px_wide: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    backtest_params: dict[str, object] | None = None,
    name: str | None = None,
    save: bool = True,
    out_dir: Path | None = None
) -> tuple[StrategyResult, dict]:
    """
    Orchestrate a Top-N backtest from a precomputed predictions/signal table.

    Validates calendars and Top-N feasibility, executes the Top-N strategy
    via `mlbt.backtest_topn.backtest_topn`, computes core metrics, and
    (optionally) persists artifacts under a timestamped run folder.

    Parameters
    ----------
    px_wide : pandas.DataFrame
        Daily prices (DatetimeIndex, columns = tickers).

    predictions : pandas.DataFrame
        MultiIndex (month, ticker) table with at least `rank_col` present.

    backtest_params : dict, optional
        Parameters for the backtesting engine, e.g. {"rank_col": y_pred", "N": 10, "cost_bps": 4.0, "strict": True}.

    name : str, optional
        Friendly label that will be included in saved metadata and filenames.

    save : bool, default=True
        If True, persist artifacts (preds/equity/weights/turnover/metrics/meta).

    out_dir : Path, optional
        Root directory where the run folder will be created.

    Returns
    -------
    (res, meta) : Tuple[StrategyResult, dict]
        `res`: StrategyResult from the backtester.
        `meta`: run metadata including run_id, paths, parameters, and small
        diagnostics (e.g., calendar coverage, universe size).
    """
    # some guards
    validate_month_grid_index(predictions)
    
    # run actual topn backtest
    backtest_params = {} if backtest_params is None else backtest_params
    res, bt_params = backtest_topn(
        px_wide=px_wide,
        predictions=predictions,
        **backtest_params,
        name=name
    )

    # compute metrics
    try:
        m = res.compute_metrics().to_dict()
    except Exception:
        m = None

    # meta data
    run_meta = build_run_meta(
        predictions=predictions,
        res=res,
        name=name,
        backtest_params=bt_params,
        runner_name="topn_from_predictions",
        runner_version=None,
        metrics=m if isinstance(m, dict) else None,
    )

    if save:
        run_dir, saved_meta = save_backtest_runs(
            run_meta=run_meta,
            hashing_params=run_meta["backtest_params"],
            res=res,
            preds=predictions,
            metrics=m if isinstance(m, dict) else None,
            base_out_dir=out_dir,
        )
        run_meta = saved_meta

    return res, run_meta