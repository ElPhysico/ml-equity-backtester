# src/mlbt/pipelines/wrappers.py
"""
This module contains convenience wrappers for our pipelines.
"""
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Tuple

from mlbt.io import read_yaml
from mlbt.utils import find_project_root, validate_config, bind_config, clean_dict, validate_px_wide_range
from mlbt.log_utils import setup_logging

from mlbt.universe import load_universe
from mlbt.load_prices import load_prices
from mlbt.calendar import build_month_end_grid
from mlbt.pipelines.ml_elasticnet_topn import run_elasticnet_topn_v0
from mlbt.strategy_result import StrategyResult
from mlbt.backtest_engines import backtest_bh
from mlbt.visualisation import plot_equities

PROJECT_ROOT = find_project_root()


def run_demo_elasticnet_topn(
    config_path: Path = PROJECT_ROOT / "config/DEMO15_config.yaml"
) -> Tuple["StrategyResult", Dict]:
    """
    Runs the ElasticNet top-N strategy pipeline. Options are controlled via config file.
    """
    setup_logging()
    # load config file containing all necessary information
    loaded_cfg = read_yaml(config_path)
    required = [
        "data_start",
        "data_end",
        "universe_file"
    ]
    if loaded_cfg is None:
        raise ValueError(f"Config is empty, requires {required} to run.")
    validate_config(loaded_cfg, required) # making sure all required keywords exist

    # observation/data period
    data_start = loaded_cfg["data_start"]
    data_end = loaded_cfg["data_end"]
    if pd.Timestamp(data_start) > pd.Timestamp(data_end):
        raise ValueError(f"data_start ({data_start}) must be earlier than data_end ({data_end}).")

    # load universe
    universe_file = loaded_cfg["universe_file"]
    tickers = load_universe(PROJECT_ROOT / "config" / "universes" / universe_file)
    if not tickers:
        raise ValueError("Ticker universe is empty.")

    # load optional parameters and safeguard
    cfg = bind_config(run_elasticnet_topn_v0, loaded_cfg)
    allowed = {
        "feature_params": {"P", "skip", "min_obs", "annualize", "ddof"},
        "label_params": {},
        "model_params": {"alpha", "l1_ratio", "random_state", "min_train_samples"},
        "backtest_params": {"rank_col", "N", "cost_bps"}
    }
    for k, v in allowed.items():
        if k in cfg:
            cfg[k] = clean_dict(cfg[k], v)

    # some range and type checks
    k = "feature_params"
    if k in cfg:
        if "P" in cfg[k]:
            cfg[k]["P"] = int(cfg[k]["P"])
        if "skip" in cfg[k]:
            cfg[k]["skip"] = int(cfg[k]["skip"])
        if "P" in cfg[k] and cfg[k]["P"] < 1:
            raise ValueError(f"'P' must be >= 1, is {cfg[k]['P']}")
        if "skip" in cfg[k] and cfg[k]["skip"] < 0:
            raise ValueError(f"'skip' must be >= 0, is {cfg[k]['skip']}")
        if "P" in cfg[k] and "skip" in cfg[k] and cfg[k]["P"] <= cfg[k]["skip"]:
            raise ValueError(f"'P' ({cfg[k]['P']}) must be > 'skip' ({cfg[k]['skip']})")
        
        if "min_obs" in cfg[k]:
            cfg[k]["min_obs"] = int(cfg[k]["min_obs"])
        if "annualize" in cfg[k]:
            if not isinstance(cfg[k]["annualize"], bool):
                raise ValueError(f"'annualize' must be type bool, is {type(cfg[k]['annualize'])}")
        if "ddof" in cfg[k]:
            cfg[k]["ddof"] = int(cfg[k]["ddof"])
        if "ddof" in cfg[k] and (cfg[k]["ddof"] != 0 and cfg[k]["ddof"] != 1):
            raise ValueError(f"'ddof' must be 0 or 1, is {cfg[k]['ddof']}")
    
    k = "model_params"
    if k in cfg:
        if "alpha" in cfg[k]:
            cfg[k]["alpha"] = float(cfg[k]["alpha"])
        if "l1_ratio" in cfg[k]:
            cfg[k]["l1_ratio"] = float(cfg[k]["l1_ratio"])
        if "random_state" in cfg[k]:
            cfg[k]["random_state"] = int(cfg[k]["random_state"])
        if "min_train_samples" in cfg[k]:
            cfg[k]["min_train_samples"] = int(cfg[k]["min_train_samples"])
        if "alpha" in cfg[k] and cfg[k]["alpha"] < 0:
            raise ValueError(f"'alpha' must be >= 0, is {cfg[k]['alpha']}")
        if "l1_ratio" in cfg[k] and (cfg[k]["l1_ratio"] < 0 or cfg[k]["l1_ratio"] > 1):
            raise ValueError(f"'l1_ratio' must be in [0, 1], is {cfg[k]['l1_ratio']}")
        
    k = "backtest_params"
    if k in cfg:
        if "N" in cfg[k]:
            cfg[k]["N"] = int(cfg[k]["N"])
        if "cost_bps" in cfg[k]:
            cfg[k]["cost_bps"] = float(cfg[k]["cost_bps"])
        if "N" in cfg[k] and cfg[k]["N"] <= 0:
            raise ValueError(f"'N' must be > 0, is {cfg[k]['N']}")
        if "cost_bps" in cfg[k] and cfg[k]["cost_bps"] < 0:
            raise ValueError(f"'cost_bps' must be >= 0, is {cfg[k]['cost_bps']}")
        

    # building correct path
    if "out_dir" in cfg:
        cfg["out_dir"] = PROJECT_ROOT / cfg["out_dir"]

    # load equity data and build canonical BME calendar grid from equity data
    px_wide = load_prices(
        in_dir=PROJECT_ROOT / "data/equity_data/",
        tickers=tickers,
        start=data_start,
        end=data_end
    )    
    me_grid = build_month_end_grid(px_wide)
    validate_px_wide_range(px_wide, data_start, data_end)
    missing = [c for c in tickers if c not in px_wide.columns]
    if missing:
        raise ValueError(f"Ticker(s) {missing} not found.")

    # run pipeline
    if "run_name" in cfg:
        run_name = cfg["run_name"]
        cfg.pop("run_name")
    res, meta = run_elasticnet_topn_v0(
        px_wide=px_wide,
        month_grid=me_grid,
        **cfg,
        run_name= "ElasticNet_" + run_name 
    )

    # load benchmarks if available
    strat_start = meta["strategy"]["strategy_start"]
    strat_end = meta["strategy"]["strategy_end"]
    bench_results = []
    if "benchmarks" in loaded_cfg and loaded_cfg["benchmarks"] is not None:
        benchmarks = set(loaded_cfg["benchmarks"])
    else:
        benchmarks = set()
    if "BH_EW_self" in benchmarks:
        b_res, b_params = backtest_bh(px_wide[(px_wide.index >= strat_start) & (px_wide.index <= strat_end)], name="BH_EW_" + run_name)
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
        except Exception as e:
            logging.warning(f"Loading benchmark prices raised an error: {e}")
        missing = [c for c in benchmarks if c not in px_wide_benchmarks.columns]
        if missing:
            logging.warning(f"Benchmark(s) {missing} not found.")
            benchmarks = benchmarks - set(missing)
        for bench in benchmarks:
            b_res, b_params = backtest_bh(px_wide_benchmarks[bench].to_frame(), name="BH_EW_"+bench)
            bench_results.append(b_res)
    
    # additional outputs
    save = cfg["save"] if "save" in cfg else False
    out_dir = PROJECT_ROOT / meta["paths"]["run_dir"] if save else None
    plot_equities([res, *bench_results], save=save, out_dir=out_dir / "benchmarks", out_name="overlay.png")
    if save:
        for br in bench_results:
            eq = br.equity.copy()
            eq.to_csv(out_dir / "benchmarks" / f"{eq.name}.csv", header=True)

    res_metrics = res.compute_metrics()
    # prepare log string
    if save:
        logging.info(f"Run ID: {meta['run_id']}")

    logging.info(f"Strategy start: {strat_start}, end: {strat_end}")

    logging.info(f"Top-N selected: {meta["backtest_params"]["N"]} | cost_bps: {meta["backtest_params"]["cost_bps"]}")

    logging.info("Metrics | " + res_metrics.to_string() + f" | Ann. avg. turnover: {100*res.ann_turnover:.2f}%")

    # comparison log string
    if bench_results:
        comparison = "Performance vs"
        for br in bench_results:
            br_metrics = br.compute_metrics()
            delta_cagr = res_metrics.cagr - br_metrics.cagr
            delta_sharpe = res_metrics.sharpe - br_metrics.sharpe
            comparison += f" | [{br.name}] {100*delta_cagr:.2f}% CAGR, {delta_sharpe:.2f} Sharpe"
        logging.info(comparison)    

    if save:
        logging.info(f"Output files saved to {meta['paths']['run_dir']}")

    return res, meta