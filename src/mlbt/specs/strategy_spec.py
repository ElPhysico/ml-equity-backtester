# src/mlbt/specs/strategy_spec.py
"""
Module to define strategy specifications.
"""
from dataclasses import dataclass, field
from collections.abc import Callable, Mapping
from mlbt.specs.strategy_result import StrategyResult


@dataclass
class StrategySpec:
    key: str
    cls: str # benchmark, ML, simple
    name: str
    runner: Callable[..., tuple[StrategyResult, dict[str, object]]]
    strat_params: Mapping[str, object] = field(default_factory=dict)
    provides_window: bool = False
    align_to: str = ""  # maybe used in the future to align one strategy to another via key