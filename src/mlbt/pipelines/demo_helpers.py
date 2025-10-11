# src/mlbt/pipelines/demo_helpers.py
"""
Convenience functions for running demos.
"""
from typing import Any
import pandas as pd
import logging
import json

from mlbt.backtest_engines import backtest_bh, backtest_mr
from mlbt.load_prices import load_prices
from mlbt.utils import find_project_root, validate_px_wide_range
from mlbt.strategy_result import StrategyResult
from mlbt.visualisation import plot_equities

PROJECT_ROOT = find_project_root()


def load_benchmarks(
    px_wide: pd.DataFrame,
    strat_start: str,
    strat_end: str,
    loaded_cfg: dict[str, Any],
    run_name: str
) -> list[StrategyResult]:
    kwargs = {}
    if "backtest_params" in loaded_cfg and loaded_cfg["backtest_params"] is not None:
        if "cost_bps" in loaded_cfg["backtest_params"]:
            kwargs["cost_bps"] = float(loaded_cfg["backtest_params"]["cost_bps"])
    
    bench_results = []
    if "benchmarks" in loaded_cfg and loaded_cfg["benchmarks"] is not None:
        benchmarks = set(loaded_cfg["benchmarks"])
    else:
        benchmarks = set()

    if "MR_EW_self" in benchmarks:
        b_res, b_params = backtest_mr(px_wide[(px_wide.index >= strat_start) & (px_wide.index <= strat_end)], **kwargs, name="MR_EW_" + run_name)
        bench_results.append(b_res)
        benchmarks.remove("MR_EW_self")

    if "BH_EW_self" in benchmarks:
        b_res, b_params = backtest_bh(px_wide[(px_wide.index >= strat_start) & (px_wide.index <= strat_end)], **kwargs, name="BH_EW_" + run_name)
        bench_results.append(b_res)
        benchmarks.remove("BH_EW_self")

    if benchmarks:
        px_wide_benchmarks = load_prices(
            in_dir=PROJECT_ROOT / "data/equity_data/",
            tickers=benchmarks,
            start=strat_start,
            end=strat_end
        )
        try:
            validate_px_wide_range(px_wide_benchmarks, strat_start, strat_end)
            # if successfully covers range, check if benchmarks are missing
            missing = [c for c in benchmarks if c not in px_wide_benchmarks.columns]
            if missing:
                logging.warning(f"Benchmark(s) {missing} not found.")
                benchmarks = benchmarks - set(missing)
        except Exception as e:
            logging.warning(f"Loading benchmark prices raised an error: {e}")
            benchmarks = set()
        
        # check if any of the benchmarks does not have data at start
        if benchmarks:
            nan_cols = px_wide_benchmarks.columns[px_wide_benchmarks.iloc[0].isna()]
            if not nan_cols.empty:
                logging.warning(f"Benchmarks with missing data are removed: {nan_cols.tolist()}")
            benchmarks = benchmarks - set(nan_cols)

        # now run BH strategy for remaining benchmarks
        for bench in benchmarks:
            b_res, b_params = backtest_bh(px_wide_benchmarks[bench].to_frame(), **kwargs, name="BH_"+bench)
            bench_results.append(b_res)

    return bench_results



def demo_additional_outputs(
    cfg: dict[str, Any],
    res: StrategyResult,
    meta: dict[str, Any],
    bench_results: list[StrategyResult],   
) -> dict[str, Any]:
    meta = meta.copy()
    strat_start = meta["strategy"]["strategy_start"]
    strat_end = meta["strategy"]["strategy_end"]

    save = cfg["save"] if "save" in cfg else False
    out_dir = PROJECT_ROOT / meta["paths"]["run_dir"] / "benchmarks" if save else None
    plot_equities([res, *bench_results], save=save, out_dir=out_dir, out_name="overlay.png")
    if save:
        for br in bench_results:
            eq = br.equity.copy()
            eq.to_csv(out_dir / f"{eq.name}.csv", header=True)

    res_metrics = res.compute_metrics()
    # prepare log string
    if save:
        logging.info(f"Run ID: {meta['run_id']}")

    logging.info(f"Strategy start: {strat_start}, end: {strat_end}")

    logging.info(f"Top-N selected: {meta["backtest_params"]["N"]} | cost_bps: {meta["backtest_params"]["cost_bps"]}")

    logging.info("Metrics | " + res_metrics.to_string() + f" | Ann. avg. turnover: {100*res.ann_turnover:.2f}%")

    # comparison log string
    if bench_results:
        meta["benchmarks"] = {}
        comparison = "Performance vs"
        for br in bench_results:
            br_metrics = br.compute_metrics()
            meta["benchmarks"][br.name] = br_metrics.to_dict()
            delta_cagr = res_metrics.cagr - br_metrics.cagr
            delta_sharpe = res_metrics.sharpe - br_metrics.sharpe
            comparison += f" | [{br.name}] {100*delta_cagr:.2f}% CAGR, {delta_sharpe:.2f} Sharpe"
        logging.info(comparison)    

    if save:
        logging.info(f"Output files saved to {meta['paths']['run_dir']}")
        p_meta = PROJECT_ROOT / meta["paths"]["meta_json"]
        p_meta.write_text(json.dumps(meta, indent=2))

    return meta