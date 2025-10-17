# src/mlbt/notebooks/runners.py
"""
This module holds convenience functions to execute whole pipelines as a single line of code in demonstration notebooks.
"""

def run_gbm_enetv0_single(
    gbm_start: str = "2000-01-01",
    gbm_end: str = "2009-12-31",
    n_runs: int = 30,
    plot: bool = True,
    master_seed: int = 1234567890,
    verbose=False 
) -> str:
    from mlbt.specs.universe_spec import univ_gbm_growth_bull
    universe_registry = {
        univ_gbm_growth_bull.key: univ_gbm_growth_bull,
    }

    from mlbt.specs.strategy_spec import strat_enet_10_v0, strat_enet_20_v0, strat_enet_50_v0
    strategy_registry = {
        strat_enet_10_v0.key: strat_enet_10_v0,
        strat_enet_20_v0.key: strat_enet_20_v0,
        strat_enet_50_v0.key: strat_enet_50_v0
    }

    from mlbt.specs.strategy_spec import bench_buy_and_hold_EW, bench_monthly_rebalance_EW
    benchmark_registry = {
        bench_buy_and_hold_EW.key: bench_buy_and_hold_EW,
        bench_monthly_rebalance_EW.key: bench_monthly_rebalance_EW
    }

    from mlbt.pipelines.full_pipeline import run_full_pipeline
    return_string = run_full_pipeline(
        sim_start=gbm_start,
        sim_end=gbm_end,
        universe_registry=universe_registry,
        strategy_registry=strategy_registry,
        benchmark_registry=benchmark_registry,
        n_runs=n_runs,
        plot=plot,
        master_seed=master_seed,
        verbose=verbose
    )

    return return_string



def test_runner(
    gbm_start: str = "2000-01-01",
    gbm_end: str = "2009-12-31",
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
