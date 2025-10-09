# src/mlbt/pipelines/wrappers.py
"""
This module contains convenience wrappers for our pipelines.
"""
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Tuple

from mlbt.io import read_yaml
from mlbt.utils import find_project_root, validate_config, bind_config, clean_dict
from mlbt.log_utils import setup_logging

from mlbt.universe import load_universe
from mlbt.load_prices import load_prices
from mlbt.calendar import build_month_end_grid
from mlbt.pipelines.ml_elasticnet_topn import run_elasticnet_topn_v0
from mlbt.strategies import StrategyResult

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
    if px_wide.empty:
        raise ValueError("Daily prices table is empty.")
    me_grid = build_month_end_grid(px_wide)
    if px_wide.index.min() > pd.Timestamp(data_start):
        raise ValueError(f"data_start ({data_start}) is earlier than earliest record in daily prices table ({px_wide.index.min()})")
    if px_wide.index.max() < pd.Timestamp(data_end):
        raise ValueError(f"data_end ({data_end}) is later than latest record in daily prices table ({px_wide.index.max()})")

    # run pipeline
    res, meta = run_elasticnet_topn_v0(
        px_wide=px_wide,
        month_grid=me_grid,
        **cfg
    )

    # prepare log string
    log_string = ""
    if "save" in cfg and cfg["save"]:
        log_string += f"Run ID: {meta['run_id']} | "
    log_string += res.compute_metrics().to_string() + f" | Avg. ann. turnover: {100*res.ann_turnover:.2f}%"
    logging.info(log_string)
    if "save" in cfg and cfg["save"]:
        logging.info(f"Output files saved to {meta['paths']['run_dir']}")

    return res, meta