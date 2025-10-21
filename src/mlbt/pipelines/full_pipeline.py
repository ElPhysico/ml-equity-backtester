# src/mlbt/pipelines/full_gbm_pipeline.py
"""
This module holds a full pipeline for creating GBM universes, evaluating strategies and benchmarks on the universe, and analysing the results.
"""
import numpy as np
import pandas as pd
import logging

from mlbt.specs.universe_spec import UniverseSpec
from mlbt.specs.strategy_spec import StrategySpec
from mlbt.calendar import build_month_end_grid, years_spanned
from mlbt.simulator.simulators import simulate_universe
from mlbt.analysis.strategy_analysis import statistics_ann_log_growth, average_metricsresults
from mlbt.log_utils import setup_logging

from mlbt.statistics.welford_aggregator import Welford


def run_full_pipeline(
    *,
    sim_start: str,
    sim_end: str,
    universe_registry: dict[str, UniverseSpec],
    strategy_registry: dict[str, StrategySpec],
    benchmark_registry: dict[str, StrategySpec],
    n_runs: int = 30,
    master_seed: int = 1234567890,
    verbose: bool = False
) -> dict[str, object]:
    """
    This pipeline simulates the GBM universe defined in 'gbm_registry', evaluates strategies and benchmarks on the universe, and computes statistics on that evaluation.

    Parameters
    ----------
    sim_start : str
        Date in YYYY-MM-DD format marking the start of the simulation horizon.

    sim_end : str
        Date in YYYY-MM-DD format marking the end of the simulation horizon.

    gbm_registry : dict[str, object]
        Registry containing the GBM universe.

    strategy_registry : dict[str, object]
        Registry containing the strategies.

    benchmark_registry : dict[str, object]
        Registry containing the benchmarks.
    
    n_runs : int, default=30
        Number of independent runs.

    master_seed : int, default=1234567890
        Seeds the main Numpy Generator.

    verbose : bool, default=False
        Toggles progress logging.

    Returns
    -------
    result : dict[str, object]
        Dictionary containing the investment horizon in years, the geometric mean equity curve as tuple containing (curve, lower band, upper band), the mean metrics dataframe, the delta log growth dataframe, and a delta log growth statistics dataframe.
    """
    setup_logging(verbose=verbose)

    # initialize RNG
    ss_master = np.random.SeedSequence(master_seed)
    run_seqs = ss_master.spawn(n_runs)

    # simulation calendar
    sim_cal = pd.bdate_range(sim_start, sim_end, freq="B")

    # collects metrics dict from MetricsResult class
    metrics = {k: [] for k in list(strategy_registry.keys()) + list(benchmark_registry.keys())}

    # Welford-online collect for log-equity curves
    log_equity = {v.name: Welford() for v in list(strategy_registry.values()) + list(benchmark_registry.values())}

    # collects the selected tickers
    selections = {k: [] for k in list(strategy_registry.keys())}

    # Welford-online collect for training coefficients
    train_coefs = {v.name: Welford() for v in list(strategy_registry.values())}

    # collects one-factor universe correlaton for checks
    corr = []

    # flag for once-per-run actions
    initial_run = True

    # run main loop
    for run_idx, ss_run in enumerate(run_seqs):
        # logic to create universe
        univ_seqs = ss_run.spawn(len(universe_registry))
        pxs = []
        for (k, u), ss_u in zip(universe_registry.items(), univ_seqs):
            u = universe_registry[k]
            rng = np.random.default_rng(ss_u)
            px, meta = simulate_universe(
                univ=u,
                calendar=sim_cal,
                rng=rng
            )
            pxs.append(px)
        px_universe = pd.concat(pxs, axis=1)

        # correlation
        corr_matrix = px_universe.pct_change().corr().values
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        corr_matrix = corr_matrix[mask]
        corr.append(np.nanmean(corr_matrix))
        if not np.isnan(corr[-1]) and abs(corr[-1]) > 0.02:
            logging.warning(f"[run {run_idx}] mean pairwise corr is {corr[-1]:.4f} (|rho|>0.02)")

        if initial_run:
            me_grid = build_month_end_grid(px_universe)      


        # logic to run strategies
        for k, s in strategy_registry.items():
            runner = s.runner
            res, meta = runner(
                px_wide=px_universe,
                month_grid=me_grid,                
                **s.strat_params,
                name=s.name
            )

            # aligning horizon
            if initial_run:
                strat_start = meta["strategy"]["strategy_start"]
                strat_end = meta["strategy"]["strategy_end"]
            if strat_start != meta["strategy"]["strategy_start"]:
                logging.warning(f"Strategies have different starting dates, benchmarks will be run over last strategy's horizon.")
            if strat_end != meta["strategy"]["strategy_end"]:
                logging.warning(f"Strategies have different ending dates, benchmarks will be run over last strategy's horizon.")
            px_strat = px_universe[(px_universe.index >= strat_start) & (px_universe.index <= strat_end)]

            # selection data
            d = res.selections
            timestamps = list(d.keys())
            changed = [t for prev, t in zip(timestamps, timestamps[1:]) if set(d[prev]) != set(d[t])]
            selections[k].append({
                "n_changes": len(changed),
                "last_change": changed[-1] if changed else None,
                "selections": d
            })

            # collecting metrics
            metrics[k].append(res.compute_metrics())

            # collecting log-equity
            eq = res.equity.to_numpy()
            eq = eq / eq[0]
            x = np.log(eq)
            log_equity[s.name].update(x)

            # collecting coefs
            coefs = meta["predictions_meta"]["training"]["coefs_by_month"]
            coefs = pd.DataFrame.from_dict(coefs).to_numpy()
            train_coefs[s.name].update(coefs)


        # logic to run benchmarks on investmentent horizon
        for k, b in benchmark_registry.items():
            runner = b.runner
            res, meta = runner(
                px_wide=px_strat,
                **b.strat_params,
                name=b.name
            )

            # collecting metrics
            metrics[k].append(res.compute_metrics())

            # collecting log-equity
            eq = res.equity.to_numpy()
            eq = eq / eq[0]
            x = np.log(eq)
            log_equity[b.name].update(x)
        
        initial_run = False


        if verbose:
            if (run_idx + 1) % max(1, n_runs // 10) == 0 or (run_idx + 1) == n_runs:
                logging.info(f"Progress {(run_idx + 1) / n_runs:.0%} ({run_idx+1}/{n_runs})")

    # years spanned by actual investment horizon
    T = years_spanned(px[(px.index >= strat_start) & (px.index <= strat_end)].index)

    # logging if verbose is set 
    logging.info(f"Number of universe constitutents: {len(px_universe.columns)}")
    logging.info(f"Mean one-factor universe correlation: {np.mean(corr):.5f}")
    logging.info(f"Investment horizon in years: {T:.2f}")

    # geometric mean log equity
    g = {k: pd.Series(np.exp(v.mean), index=px_strat.index, name=k) for k,v in log_equity.items()}
    l = {k: pd.Series(np.exp(v.ci95_typical[0]), index=px_strat.index, name=k) for k,v in log_equity.items()}
    h = {k: pd.Series(np.exp(v.ci95_typical[1]), index=px_strat.index, name=k) for k,v in log_equity.items()}

    # statistics for delta log-growth
    delta_log_growth = []
    for ks, s in strategy_registry.items():
        for kb, b in benchmark_registry.items():
            for i, sm in enumerate(metrics[ks]):
                bm = metrics[kb][i]
                delta_log_growth.append({
                    "strategy": s.name,
                    "benchmark": b.name,
                    "ann_log_growth": (np.log1p(sm.total_return) - np.log1p(bm.total_return)) / T
                })
    delta_log_growth_df = pd.DataFrame(delta_log_growth)
    delta_log_growth_stats = (
        delta_log_growth_df.groupby(["strategy", "benchmark"])["ann_log_growth"]
        .apply(statistics_ann_log_growth, alt="less", cl=0.95)
        .unstack().reset_index()
    )

    # strategy metrics statistics
    name_map = {k: (strategy_registry.get(k) or benchmark_registry.get(k)).name for k in metrics.keys()}
    mean_metrics_df = pd.DataFrame({name_map[k]: average_metricsresults(v, bootstrap_vol=True) for k, v in metrics.items()}).T
    full_metrics_df = pd.concat(
        {
            k: (pd.DataFrame([m.to_dict() for m in v])
                .drop(columns=['freq'], errors='ignore')
                .where(lambda df: df.notna(), other=0.0)) # needed bc turnover is None for Buy and hold
            for k, v in metrics.items()
        },
        names=['strategy', 'sample']
    )
    ns = [v.strat_params["backtest_params"]["N"] for v in strategy_registry.values()]
    ks = [len(px_universe.columns) for _ in benchmark_registry.values()]
    mean_metrics_df["n_tickers"] = ns + ks
    

    results = {
        "years": T,
        "geometric_mean_equity": (g, (l, h)),
        "mean_metrics": mean_metrics_df,
        "delta_log_growth": delta_log_growth_df,
        "delta_log_growth_stats": delta_log_growth_stats,
        "selections": selections,
        "train_coefs": train_coefs
    }

    return results