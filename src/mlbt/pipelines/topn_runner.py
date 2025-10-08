# src/mlbt/pipelines/topn_runner.py
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

from mlbt.strategies import StrategyResult
from mlbt.backtest_topn import backtest_topn
from mlbt.utils import build_run_meta
from mlbt.io import save_backtest_runs


def run_topn_from_predictions(
    px_wide: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    rank_col: str,
    N: int = 10,
    cost_bps: float = 5.0,
    run_name: Optional[str] = None,
    strict: bool = True,
    save: bool = True,
    out_dir: Optional[Path] = None
) -> Tuple["StrategyResult", Dict]:
    """
    Orchestrate a Top-N backtest from a precomputed predictions table.

    Validates calendars and Top-N feasibility, executes the Top-N strategy
    via `mlbt.backtest_topn.backtest_topn`, computes core metrics, and
    (optionally) persists artifacts under a timestamped run folder.

    Parameters
    ----------
    px_wide : pandas.DataFrame
        Daily prices (DatetimeIndex, columns = tickers).

    predictions : pandas.DataFrame
        MultiIndex (month, ticker) table with at least `rank_col` present.

    rank_col : str
        Column inside `predictions` used to rank tickers each month (higher is better).

    N : int, default=10
        Number of tickers to select each rebalance.

    cost_bps : float, default=5.0
        One-way transaction cost (basis points) applied on turnover at rebalances.

    run_name : str, optional
        Friendly label that will be included in saved metadata and filenames.

    strict : bool, default=True
        If True, raise on missing prediction months or under-filled Top-N months.
        If False, a future variant may apply a fallback policy (carry/to_cash/skip).

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

    Notes
    -----
    - This runner does not build features or train models; it assumes your
      `predictions` are already prepared and aligned to (month, ticker).
    - All heavy lifting of execution happens in `backtest_topn`.
    """
    # checking for ranking col
    if rank_col not in predictions.columns:
        raise ValueError(f"`rank_col='{rank_col}'` not found in predictions columns: {list(predictions.columns)}")
    
    # sanity on index level
    if not isinstance(predictions.index, pd.MultiIndex):
        raise ValueError("`predictions` must be indexed by MultiIndex (month, ticker).")
    if "month" not in predictions.index.names or "ticker" not in predictions.index.names:
        raise ValueError("`predictions` index must have levels ['month','ticker'].")
    
    # run actual topn backtest
    res = backtest_topn(
        px_wide=px_wide,
        predictions=predictions,
        rank_col=rank_col,
        N=N,
        cost_bps=cost_bps,
        name=run_name
    )

    # compute metrics
    try:
        m = res.compute_metrics()
    except Exception:
        m = None

    # meta data
    params = {
        "rank_col": rank_col,
        "N": N,
        "cost_bps": cost_bps,
        "strict": strict
    }

    run_meta = build_run_meta(
        predictions=predictions,
        res=res,
        run_name=run_name,
        runner_name="topn_from_predictions",
        runner_version="v0",
        params=params,
        metrics=m if isinstance(m, dict) else None,
    )

    if save:
        run_dir, saved_meta = save_backtest_runs(
            run_meta=run_meta,
            params=run_meta["params"],
            res=res,
            preds=predictions,
            metrics=m if isinstance(m, dict) else None,
            base_out_dir=out_dir,
        )
        run_meta = saved_meta

    return res, run_meta