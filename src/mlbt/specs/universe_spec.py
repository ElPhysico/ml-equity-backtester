# src/mlbt/spec/universe_spec.py
"""
Module to define universe specifications.
"""
from dataclasses import dataclass, field
from collections.abc import Callable, Mapping
import pandas as pd


@dataclass
class UniverseSpec:
    key: str
    cls: str
    sector: str
    name: str
    simulator: Callable[..., tuple[pd.DataFrame, dict[str, object]]]
    n_tickers: int
    sim_params: Mapping[str, object] = field(default_factory=dict)
    meta: dict[str, object] = None # reserved for actual simulation meta




