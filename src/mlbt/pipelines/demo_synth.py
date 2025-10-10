# src/mlbt/pipelines/demo_synth.py
"""
This module contains the demo runner using synthetic maret data.
"""
import logging
from pathlib import Path
import pandas as pd

from mlbt.log_utils import setup_logging
from mlbt.utils import find_project_root, validate_config, bind_config, validate_px_wide_range
from mlbt.io import read_yaml

from mlbt.simulator.trajectories import simulate_gbm_trajectories
from mlbt.simulator.validaton import validate_gbm_scaling_and_drift

from mlbt.pipelines.ml_elasticnet_topn import run_elasticnet_topn_v0
from mlbt.pipelines.demo_validation import demo_validate_v0
from mlbt.pipelines.demo_helpers import load_benchmarks, demo_additional_outputs
from mlbt.calendar import build_month_end_grid
from mlbt.strategy_result import StrategyResult

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
    sim_kwargs = {}
    if "seed" in loaded_cfg["simulator_params"]:
        sim_kwargs["seed"] = int(loaded_cfg["simulator_params"]["seed"])

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
        if "ann_mu" in gbm_cfg:
            gbm_cfg["ann_mu"] = float(gbm_cfg["ann_mu"])
        if "ann_sigma" in gbm_cfg:
            gbm_cfg["ann_sigma"] = float(gbm_cfg["ann_sigma"])
        if "ann_sigma" in gbm_cfg and gbm_cfg["ann_sigma"] < 0:
            raise ValueError(f"'ann_sigma' needs to be >= 0, is {gbm_cfg['ann_sigma']}")
        if "tdy" in gbm_cfg:
            gbm_cfg["tdy"] = int(gbm_cfg["tdy"])
        if "tdy" in gbm_cfg and gbm_cfg["tdy"] < 0:
            raise ValueError(f"'tdy' needs to be >= 0, is {gbm_cfg['tdy']}")
        
        gbm_px, gbm_meta = simulate_gbm_trajectories(
            calendar=calendar,
            **gbm_cfg,
            **sim_kwargs
        )
        report = validate_gbm_scaling_and_drift(
            px_wide=gbm_px,
            ann_mu=gbm_meta["params"]["ann_mu"],
            ann_sigma=gbm_meta["params"]["ann_sigma"]
        )
        if not all(report["pass"].values()):
            logging.warning(f"GBM universe didn't pass check: {report['pass']}")
        
        universes.append(gbm_px)
        universe_metas["gbm"] = gbm_meta
        

    # finished loading all universes
    if not universes:
        raise ValueError("Universe is empty.")
    
    px_wide = pd.concat(universes, axis=1)
    me_grid = build_month_end_grid(px_wide)
    validate_px_wide_range(px_wide, start, end)

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

    # adding any potential universe creation parameters to metadata
    if universe_metas:
        meta["universe_simulations"] = universe_metas

    # additional outputs
    demo_additional_outputs(
        cfg=cfg,
        res=res,
        meta=meta,
        bench_results=bench_results
    )

    return res, meta