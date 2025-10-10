# src/mlbt/pipelines/demo_validation.py
"""
Module contains helper functions to validate inputs allowed for demos.
"""
from typing import Any

from mlbt.utils import clean_dict


def demo_validate_v0(
    cfg: dict[str, Any]
) -> dict[str, Any]:
    cfg = cfg.copy()
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
        
    return cfg