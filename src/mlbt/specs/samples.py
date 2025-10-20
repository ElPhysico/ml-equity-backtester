# src/mlbt/specs/samples.py
"""
Contains sample universes, strategies, and benchmarks.
"""
from dataclasses import replace

from mlbt.specs.universe_spec import UniverseSpec
from mlbt.specs.strategy_spec import StrategySpec


# some ready to use universes
from mlbt.simulator.simulators import simulate_gbm_trajectories

# Insights #####################################################################
# univ_gbm_enet_single_regime = UniverseSpec(
#     key="enet_single_regime",
#     cls="GBM",
#     sector="Insights",
#     name="Insights single",
#     simulator=simulate_gbm_trajectories,
#     n_tickers=500,
#     sim_params={
#         "mu": [0.1],
#         "sigma": [0.25]
#     }
# )


# Demo universe ################################################################
univ_gbm_demo = UniverseSpec(
    key="gbm_demo",
    cls="GBM",
    sector="Demo",
    name="GBM demo universe",
    simulator=simulate_gbm_trajectories,
    n_tickers=100,
    sim_params={
        "mu": [0.05],
        "sigma": [0.25],
        "switch_at": [],
        "labels": ["updrift medium vol"]
    }
)

# growth sector ################################################################
univ_gbm_growth_bull = UniverseSpec(
    key="growth_bull",
    cls="GBM equities",
    sector="Growth",
    name="Growth stocks",
    simulator=simulate_gbm_trajectories,
    n_tickers=100,
    sim_params={
        "mu": [0.1],
        "sigma": [0.25],
        "switch_at": [],
        "labels": ["bull"]
    }
)

univ_gbm_growth_bear = replace(
    univ_gbm_growth_bull,
    key="growth_bear",
    sim_params={
        "mu": [-0.03],
        "sigma": [0.3],
        "switch_at": [],
        "labels": ["bear"]
    }
)

univ_gbm_growth_bull_switch_bear = replace(
    univ_gbm_growth_bull,
    key="growth_bull_to_bear",
    sim_params={
        "mu": [0.1, -0.03],
        "sigma": [0.25, 0.3],
        "switch_at": [0.5],
        "labels": ["bull", "bear"]
    }
)


# value sector #################################################################
univ_gbm_value_bull = UniverseSpec(
    key="value_bull",
    cls="GBM equities",
    sector="Value",
    name="Value stocks",
    simulator=simulate_gbm_trajectories,
    n_tickers=80,
    sim_params={
        "mu": [0.07],
        "sigma": [0.15],
        "switch_at": [],
        "labels": ["bull"]
    }
)

univ_gbm_value_bear = replace(
    univ_gbm_value_bull,
    key="value_bear",
    sim_params={
        "mu": [-0.01],
        "sigma": [0.2],
        "switch_at": [],
        "labels": ["bear"]
    }
)

univ_gbm_value_bull_switch_bear = replace(
    univ_gbm_value_bull,
    key="value_bull_to_bear",
    sim_params={
        "mu": [0.07, -0.01],
        "sigma": [0.15, 0.2],
        "switch_at": [0.6],
        "labels": ["bull", "bear"]
    }
)



# a few strategies
from mlbt.pipelines.ml_elasticnet_topn import run_elasticnet_topn_v0
import copy
cost_bps = 10

strat_enet_10_v0 = StrategySpec(
    key="enet_top10_v0",
    cls="ml",
    name="ElasticNet Top-10",
    runner=run_elasticnet_topn_v0,
    strat_params={
        "feature_params": {
            "P": 12,
            "skip": 1,
            "min_obs": 10,
            "annualize": False,
            "ddof": 0
        },
        "label_params": {},
        "model_params": {
            "alpha": 0.001,
            "l1_ratio": 0.5,
            "min_train_samples": 100
        },
        "backtest_params": {
            "rank_col": "y_pred",
            "N": 10,
            "cost_bps": cost_bps
        }
    },
    provides_window=True
)

new_params = copy.deepcopy(strat_enet_10_v0.strat_params)
new_params["backtest_params"]["N"] = 5
strat_enet_5_v0 = replace(
    strat_enet_10_v0,
    key="enet_top5_v0",
    name="ElasticNet Top-5",
    strat_params=new_params
)

new_params = copy.deepcopy(strat_enet_10_v0.strat_params)
new_params["backtest_params"]["N"] = 20
strat_enet_20_v0 = replace(
    strat_enet_10_v0,
    key="enet_top20_v0",
    name="ElasticNet Top-20",
    strat_params=new_params
)

new_params = copy.deepcopy(strat_enet_10_v0.strat_params)
new_params["backtest_params"]["N"] = 50
strat_enet_50_v0 = replace(
    strat_enet_10_v0,
    key="enet_top50_v0",
    name="ElasticNet Top-50",
    strat_params=new_params
)

new_params = copy.deepcopy(strat_enet_10_v0.strat_params)
new_params["backtest_params"]["N"] = 100
strat_enet_100_v0 = replace(
    strat_enet_10_v0,
    key="enet_top100_v0",
    name="ElasticNet Top-100",
    strat_params=new_params
)




# a few benchmarks
from mlbt.backtest_engines.backtest_bh import backtest_bh
from mlbt.backtest_engines.backtest_mr import backtest_mr

bench_buy_and_hold_EW = StrategySpec(
    key="buy_and_hold_ew",
    cls="benchmark",
    name="Buy & Hold EW",
    runner=backtest_bh,
    strat_params={
        "weights": None,
        "cost_bps": cost_bps
    },
    provides_window=False
)

bench_monthly_rebalance_EW = StrategySpec(
    key="monthly_rebalance_ew",
    cls="benchmark",
    name="Monthly Rebalance EW",
    runner=backtest_mr,
    strat_params={
        "weights": None,
        "cost_bps": cost_bps
    },
    provides_window=False
)