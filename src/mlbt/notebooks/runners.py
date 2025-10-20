# src/mlbt/notebooks/runners.py
"""
This module holds convenience functions to execute whole pipelines as a single line of code in demonstration notebooks.
"""
import logging

from mlbt.log_utils import setup_logging



def run_gbm_enetv0_single_experiment(
    gbm_start: str = "2000-01-01",
    gbm_end: str = "2009-12-31",
    n_runs: int = 50,
    n_tickers: int | None = 250,
    mu: float | None = 0.1,
    sigma: float | None = 0.25,
    cost_bps: int | None = 10,
    master_seed: int = 1990,
    verbose=False 
) -> str:
    setup_logging(verbose)
    logging.info(locals())

    # import pre-defined univ, stras, benches
    from mlbt.specs.samples import (
        strat_enet_5_v0,
        strat_enet_10_v0,
        strat_enet_20_v0,
        strat_enet_50_v0,
        strat_enet_100_v0,

        bench_buy_and_hold_EW,
        bench_monthly_rebalance_EW
    )

    from mlbt.specs.universe_spec import UniverseSpec
    from mlbt.simulator.simulators import simulate_gbm_trajectories
    univ_gbm_enet_single_regime = UniverseSpec(
        key="enet_single_regime",
        cls="GBM",
        sector="Insights",
        name="Insights single",
        simulator=simulate_gbm_trajectories,
        n_tickers=n_tickers,
        sim_params={
            "mu": [mu],
            "sigma": [sigma]
        }
    )

    # sanity check, remove later
    # from mlbt.specs.strategy_spec import StrategySpec
    # from mlbt.pipelines.random_n import run_randomn
    # from dataclasses import replace
    # random_5 = StrategySpec(
    #     key="random5",
    #     cls="check",
    #     name="Random 5",
    #     runner=run_randomn,
    #     strat_params={"backtest_params": {
    #         "N": 5,
    #         "cost_bps": 10
    #     }}
    # )
    # random_10 = replace(
    #     random_5,
    #     key="random10",
    #     name="Random 10",
    #     strat_params={"backtest_params": {
    #         "N": 10,
    #         "cost_bps": 10
    #     }}
    # )
    # random_20 = replace(
    #     random_5,
    #     key="random20",
    #     name="Random 20",
    #     strat_params={"backtest_params": {
    #         "N": 20,
    #         "cost_bps": 10
    #     }}
    # )
    # random_50 = replace(
    #     random_5,
    #     key="random50",
    #     name="Random 50",
    #     strat_params={"backtest_params": {
    #         "N": 50,
    #         "cost_bps": 10
    #     }}
    # )
    # random_100 = replace(
    #     random_5,
    #     key="random100",
    #     name="Random 100",
    #     strat_params={"backtest_params": {
    #         "N": 100,
    #         "cost_bps": 10
    #     }}
    # )
    ###################

    # register
    universe_registry = {
        univ_gbm_enet_single_regime.key: univ_gbm_enet_single_regime,
    }
    strategy_registry = {
        strat_enet_5_v0.key: strat_enet_5_v0,
        strat_enet_10_v0.key: strat_enet_10_v0,
        strat_enet_20_v0.key: strat_enet_20_v0,
        strat_enet_50_v0.key: strat_enet_50_v0,
        strat_enet_100_v0.key: strat_enet_100_v0,

        # random_5.key: random_5,
        # random_10.key: random_10,
        # random_20.key: random_20,
        # random_50.key: random_50,
        # random_100.key: random_100,
    }
    benchmark_registry = {
        bench_buy_and_hold_EW.key: bench_buy_and_hold_EW,
        bench_monthly_rebalance_EW.key: bench_monthly_rebalance_EW
    }

    # overwrite costs
    if cost_bps is not None:
        for k in strategy_registry.keys():
            strategy_registry[k].strat_params["backtest_params"]["cost_bps"] = cost_bps
        for k in benchmark_registry.keys():
            benchmark_registry[k].strat_params["cost_bps"] = cost_bps

    # run pipeline
    from mlbt.pipelines.full_pipeline import run_full_pipeline
    result = run_full_pipeline(
        sim_start=gbm_start,
        sim_end=gbm_end,
        universe_registry=universe_registry,
        strategy_registry=strategy_registry,
        benchmark_registry=benchmark_registry,
        n_runs=n_runs,
        master_seed=master_seed,
        verbose=verbose
    )

    return result



def test_runner(
    gbm_start: str = "2000-01-01",
    gbm_end: str = "2019-12-31",
    n_runs: int = 30,
    plot: bool = False,
    master_seed: int = 1234567890    
) -> None:
    from mlbt.specs.universe_spec import univ_gbm_growth_bull, univ_gbm_value_bull, univ_gbm_growth_bull_switch_bear
    universe_registry = {
        univ_gbm_growth_bull.key: univ_gbm_growth_bull,
        univ_gbm_value_bull.key: univ_gbm_value_bull,
        univ_gbm_growth_bull_switch_bear.key: univ_gbm_growth_bull_switch_bear
    }

    from mlbt.specs.strategy_spec import strat_enet_10_v0, strat_enet_20_v0, bench_buy_and_hold_EW, bench_monthly_rebalance_EW
    strategy_registry = {
        strat_enet_10_v0.key: strat_enet_10_v0,
        strat_enet_20_v0.key: strat_enet_20_v0
    }
    benchmark_registry = {
        bench_buy_and_hold_EW.key: bench_buy_and_hold_EW,
        bench_monthly_rebalance_EW.key: bench_monthly_rebalance_EW
    }

    from mlbt.pipelines.full_pipeline import run_full_pipeline
    _ = run_full_pipeline(
        sim_start=gbm_start,
        sim_end=gbm_end,
        universe_registry=universe_registry,
        strategy_registry=strategy_registry,
        benchmark_registry=benchmark_registry,
        n_runs=n_runs,
        plot=plot,
        master_seed=master_seed
    )
