# src/mlbt/simulator/meta.py
"""
Module contains helpers to write meta data for simulations.
"""
import pandas as pd
from typing import Any
import json
from mlbt.utils import sha1_of_str


def build_trajectories_meta(
    n_tickers: int,
    calendar: pd.DatetimeIndex,
    params: dict[str, Any],
    seed: int
) -> dict[str, Any]:
    meta = {
        "n_tickers": n_tickers,
        "start": calendar[0].strftime("%Y-%m-%d"),
        "end": calendar[-1].strftime("%Y-%m-%d"),
        "params": params,
        "seed": seed
    }
    meta_hash = sha1_of_str(json.dumps(meta, sort_keys=True))
    meta["hash"] = meta_hash

    return meta