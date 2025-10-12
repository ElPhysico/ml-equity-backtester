# src/mlbt/visualisation.py
"""
Visualisation tools and helpers.

This module holds tools and helpers to visualise a variety of data, such as equity curves.
"""
import pandas as pd
from pathlib import Path
from collections.abc import Sequence
import matplotlib.pyplot as plt
import seaborn as sns

from mlbt.strategy_result import StrategyResult


def plot_results(
    results: Sequence[StrategyResult],
    names: Sequence[str] | None = None,
    save: bool = False,
    out_dir: Path | None = None,
    out_name: str = "equities.png"
) -> None:
    if names is None:
        names = [sr.name for sr in results]
    equities = [sr.equity for sr in results]
    df = pd.concat(equities, axis=1, keys=names).dropna(how="all")
    
    sns.set_theme(context="talk", style="whitegrid")
    fig = plt.figure(figsize=(12, 6))
    sns.lineplot(data=df)
    plt.title("Equity Curves (Rebased to 1.0 at T0)")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend(title="Strategy", loc="best")
    plt.tight_layout()
    if save:
        if out_dir is not None:
            out_dir.mkdir(exist_ok=True, parents=True)
            plt.savefig(out_dir / out_name, dpi=150)
        else:
            raise ValueError(f"Cannot save figure as 'out_dir' is not specified.")    
    # plt.show()