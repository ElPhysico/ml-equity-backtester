# /src/mlbt/pipelines/random_topn.py
import pandas as pd
import numpy as np

from mlbt.specs.strategy_result import StrategyResult
from mlbt.utils import validate_px_wide_index
from mlbt.backtest_engines.backtest_topn import backtest_topn
from mlbt.utils import build_run_meta
from mlbt.io import save_backtest_runs


def run_randomn(
    px_wide: pd.DataFrame,
    month_grid: pd.DataFrame,
    rng: np.random.Generator | None = None,
    backtest_params: dict[str, object] | None = None,
    name: dict[str, object] | None = None,
    save: bool = False,
    out_dir: dict[str, object] | None = None    
) -> tuple[StrategyResult, dict[str, object]]:
    """
    Orchestrate a Random-N backtest.

    Parameters
    ----------
    px_wide : pandas.DataFrame
        Daily prices (DatetimeIndex, columns = tickers).

    month_grid : pandas.DataFrame
        MultiIndex (month, ticker) universe grid controlling panel alignment.

    model_params : dict, optional
        Contains the numpy.random.Generator.

    backtest_params : dict, optional
        Parameters for the backtesting engine, here; {"N": 10, "cost_bps": 4.0, "strict": True}.

    name : str, optional
        Friendly label appearing in saved metadata and filenames.

    save : bool, default=False
        If True, persist predictions, backtest outputs, metrics, and metadata.

    out_dir : Path, optional
        Root directory for the run folder.

    Returns
    -------
    (res, meta) : Tuple[StrategyResult, dict]
        `res`: StrategyResult of the Random-N backtest.
        `meta`: metadata.
    """
    # some guards
    validate_px_wide_index(px_wide)

    if rng is None:
        rng = np.random.default_rng(1990)

    backtest_params = {"N": 10} if backtest_params is None else backtest_params
    preds = month_grid.groupby(level="month", group_keys=False).apply(lambda x: x.sample(backtest_params["N"], replace=False, random_state=rng))
    preds["y_pred"] = 1.0

    res, bt_params = backtest_topn(
        px_wide=px_wide,
        predictions=preds,
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
        predictions=preds,
        res=res,
        name=name,
        backtest_params=bt_params,
        runner_name="elasticnet_topn",
        runner_version="v0",
        predictions_meta=None,
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