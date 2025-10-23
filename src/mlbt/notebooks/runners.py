# src/mlbt/notebooks/runners.py
"""
This module holds convenience functions to execute whole pipelines as a single line of code in demonstration notebooks.
"""
import logging

from mlbt.log_utils import setup_logging
from mlbt.utils import normalize_to_list, broadcast_lists


def run_gbm_enetv0_experiment(
    gbm_start: str = "2000-01-01",
    gbm_end: str = "2009-12-31",
    n_runs: int = 50,
    n_tickers: int | list[int] | None = 250,
    mu: float | list[float] | None = 0.1,
    sigma: float | list[float] | None = 0.25,
    cost_bps: int | None = 10,
    master_seed: int = 1990,
    verbose=False 
) -> dict[str, object]:
    """
    Runs one of the first complete strategy experiments using the ElasticNet v0 version compared to Buy & Hold and Monthly Rebalance benchmarks in simulated GBM universes.

    Parameters
    ----------
    gbm_start : str, default="2000-01-01"
        Start of the simulated universe, this is NOT the start of the strategy, as the strategy will only start after training on an initial period.

    gbm_end : str, default="2009-12-31"
        End of the simulated universe.

    n_runs : int, default=50
        Number of independent simulations.

    n_tickers : int | list[int], default=250
        Number of tickers per universe. Accepts a single value, which initializes all universes with the same number of tickers or a list containing the number of tickers per universe. Will be broadcasted to match 'mu' and 'sigma'.

    mu : float | list[float], default=0.1
        GBM drift per universe. Accepts a single value, which initializes all universes with the same drift or a list containing the drift per universe. Will be broadcasted to match 'n_tickers' and 'sigma'.

    sigma : float | list[float], default=0.25
        GBM volatility per universe. Accepts a single value, which initializes all universes with the same volatility or a list containing the volatility per universe. Will be broadcasted to match 'n_tickers' and 'mu'.

    cost_bps : int, default=10
        The one-sided cost applied to turnover (both buy and sell) given in bps.

    master_seed : int, default=1990
        Seeds all RNG processes and can be used to create identical outcomes.

    verbose : bool, default=False
        Sets the logging.level to INFO and enables additional informational output to command line.

    Returns
    -------
    result : dict[str, object]
        Dictionary containing the investment horizon in years, the geometric mean equity curve as tuple containing (curve, lower band, upper band), the mean metrics dataframe, the delta log growth dataframe, and a delta log growth statistics dataframe. May be adjusted in future updates.
    """
    setup_logging(verbose)
    logging.info(locals())

    # Normalize
    n_tickers = normalize_to_list(n_tickers, element_type=int)
    mu        = normalize_to_list(mu,        element_type=(int, float))
    sigma     = normalize_to_list(sigma,     element_type=(int, float))

    # Broadcast to a common length (e.g., across parameter sweeps)
    n_tickers, mu, sigma = broadcast_lists(n_tickers, mu, sigma)

    # create gbm universes
    from mlbt.specs.universe_spec import UniverseSpec
    from mlbt.simulator.simulators import simulate_gbm_trajectories
    universe_registry = {}
    for i, (k, m, s) in enumerate(zip(n_tickers, mu, sigma)):
        key = f"{i:02d}_gbm_k={k}_mu={m}_sigma={s}"
        univ = UniverseSpec(
            key=key,
            cls="GBM",
            sector="Insights",
            name=f"{i:02d}_GBM_K={k}_Mu={m}_Sigma={s}",
            simulator=simulate_gbm_trajectories,
            n_tickers=k,
            sim_params={
                "mu": [m],
                "sigma": [s]
            }
        )
        universe_registry[key] = univ

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

    # register
    strategy_registry = {
        strat_enet_5_v0.key: strat_enet_5_v0,
        strat_enet_10_v0.key: strat_enet_10_v0,
        strat_enet_20_v0.key: strat_enet_20_v0,
        strat_enet_50_v0.key: strat_enet_50_v0,
        strat_enet_100_v0.key: strat_enet_100_v0,
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
