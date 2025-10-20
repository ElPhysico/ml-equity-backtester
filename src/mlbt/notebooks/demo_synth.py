# src/mlbt/pipelines/demo_synth.py
"""
This module contains the demo runner using synthetic maret data.
"""
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from mlbt.log_utils import setup_logging
from mlbt.utils import find_project_root, validate_config, bind_config, validate_px_wide_range
from mlbt.io import read_yaml

from mlbt.simulator.simulators import simulate_gbm_trajectories
from mlbt.simulator.validaton import validate_gbm_scaling_and_drift

from mlbt.pipelines.ml_elasticnet_topn import run_elasticnet_topn_v0
from mlbt.notebooks.demo_validation import demo_validate_v0
from mlbt.notebooks.demo_helpers import load_benchmarks, demo_additional_outputs
from mlbt.calendar import build_month_end_grid
from mlbt.specs.strategy_result import StrategyResult

PROJECT_ROOT = find_project_root()


def run_demo_synth_elasticnet_topn(
    config_path: Path = PROJECT_ROOT / "config/DEMO_synth_config.yaml"
) -> tuple[StrategyResult, dict]:
    """
    Creates a synthetic universe and runs the ElasticNet top-N strategy pipeline. Options are controlled via config file.
    """
    setup_logging()
    # load config file containing all necessary information
    loaded_cfg = read_yaml(config_path)    
    required = [
        "start",
        "end"
    ]
    if "simulator_params" not in loaded_cfg or loaded_cfg["simulator_params"] is None:
        raise ValueError(f"Config requires key 'simulator_params' containing {required}")
    validate_config(loaded_cfg["simulator_params"], required, log_name="simulator_params")

    # simulator period
    start = loaded_cfg["simulator_params"]["start"]
    end = loaded_cfg["simulator_params"]["end"]
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError(f"start ({start}) must be earlier than end ({end}).")
    calendar = pd.bdate_range(start, end, freq="B")

    # seed
    if "seed" in loaded_cfg["simulator_params"]:
        # sim_kwargs["seed"] = int(loaded_cfg["simulator_params"]["seed"])
        rng = np.random.default_rng(int(loaded_cfg["simulator_params"]["seed"]))
    else:
        rng = None

    # load optional parameters and safeguard
    cfg = bind_config(run_elasticnet_topn_v0, loaded_cfg)
    cfg = demo_validate_v0(cfg)

    # building correct path
    if "out_dir" in cfg:
        cfg["out_dir"] = PROJECT_ROOT / cfg["out_dir"]
    
    universes = []
    universe_metas = {}
    # create gbm universe
    if "gbm_params" in loaded_cfg and loaded_cfg["gbm_params"] is not None:
        gbm_cfg = bind_config(simulate_gbm_trajectories, loaded_cfg["gbm_params"])
        if "n_tickers" in gbm_cfg:
            gbm_cfg["n_tickers"] = int(gbm_cfg["n_tickers"])
        if "n_tickers" in gbm_cfg and gbm_cfg["n_tickers"] < 1:
            raise ValueError(f"'n_tickers' needs to be > 0, is {gbm_cfg['n_tickers']}")
        if "mu" in gbm_cfg:
            gbm_cfg["mu"] = float(gbm_cfg["mu"])
        if "sigma" in gbm_cfg:
            gbm_cfg["sigma"] = float(gbm_cfg["sigma"])
        if "sigma" in gbm_cfg and gbm_cfg["sigma"] < 0:
            raise ValueError(f"'sigma' needs to be >= 0, is {gbm_cfg['sigma']}")
        
        gbm_px, gbm_meta = simulate_gbm_trajectories(
            calendar=calendar,
            **gbm_cfg,
            rng=rng
        )
        report = validate_gbm_scaling_and_drift(
            px_wide=gbm_px,
            mu=gbm_meta["params"]["mu"][0],
            sigma=gbm_meta["params"]["sigma"][0]
        )
        if not all(report["pass"].values()):
            logging.warning(f"GBM universe didn't pass check: {report['pass']}")
            print(report)
        
        universes.append(gbm_px)
        universe_metas["gbm"] = gbm_meta
        

    # finished loading all universes
    if not universes:
        raise ValueError("Universe is empty.")
    
    px_wide = pd.concat(universes, axis=1)
    me_grid = build_month_end_grid(px_wide)
    validate_px_wide_range(px_wide, start, end)

    # run pipeline
    if "name" in cfg:
        name = cfg["name"]
        cfg.pop("name")
    res, meta = run_elasticnet_topn_v0(
        px_wide=px_wide,
        month_grid=me_grid,
        **cfg,
        name= "ElasticNet_" + name 
    )

    # load benchmarks if available
    strat_start = meta["strategy"]["strategy_start"]
    strat_end = meta["strategy"]["strategy_end"]
    bench_results = load_benchmarks(
        px_wide=px_wide,
        strat_start=strat_start,
        strat_end=strat_end,
        loaded_cfg=loaded_cfg,
        name=name
    )

    # adding any potential universe creation parameters to metadata
    if universe_metas:
        meta["universe_simulations"] = universe_metas

    # additional outputs
    meta = demo_additional_outputs(
        cfg=cfg,
        res=res,
        meta=meta,
        bench_results=bench_results
    )

    return res, meta