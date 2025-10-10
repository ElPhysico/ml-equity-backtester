# src/mlbt/pipelines/demo15.py
"""
This module contains the demo15 runner.
"""
import pandas as pd
from pathlib import Path
import logging

from mlbt.io import read_yaml
from mlbt.utils import find_project_root, validate_config, bind_config,validate_px_wide_range
from mlbt.log_utils import setup_logging

from mlbt.universe import load_universe
from mlbt.load_prices import load_prices
from mlbt.calendar import build_month_end_grid
from mlbt.pipelines.ml_elasticnet_topn import run_elasticnet_topn_v0
from mlbt.pipelines.demo_helpers import load_benchmarks, demo_additional_outputs
from mlbt.strategy_result import StrategyResult
from mlbt.visualisation import plot_equities
from mlbt.pipelines.demo_validation import demo_validate_v0

PROJECT_ROOT = find_project_root()


def run_demo15_elasticnet_topn(
    config_path: Path = PROJECT_ROOT / "config/DEMO15_config.yaml"
) -> tuple[StrategyResult, dict]:
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
    cfg = demo_validate_v0(cfg)

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
    bench_results = load_benchmarks(
        px_wide=px_wide,
        strat_start=strat_start,
        strat_end=strat_end,
        loaded_cfg=loaded_cfg,
        run_name=run_name
    )
    
    # additional outputs
    meta = demo_additional_outputs(
        cfg=cfg,
        res=res,
        meta=meta,
        bench_results=bench_results
    )

    return res, meta