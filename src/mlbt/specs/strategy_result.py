# src/mlbt/specs/result.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List
import json

import numpy as np
import pandas as pd


@dataclass
class StrategyResult:
    """
    Container for a single backtest run with diagnostics.
    """

    # Core
    name: str
    equity: pd.Series

    # Diagnostics / trading artifacts
    rebal_dates: pd.DatetimeIndex = field(default_factory=pd.DatetimeIndex)
    turnover: Optional[pd.Series] = None  # one-way ∑|w_new - w_old| per rebalance, exclude initial entry
    entry_cost_frac: float = 0.0          # e.g., cost_bps / 1e4 applied once at initial entry
    selections: Optional[Dict[pd.Timestamp, List[str]]] = None  # date -> [tickers]
    weights: Optional[pd.DataFrame] = None                      # rows=rebal_dates, cols=tickers

    # Provenance
    params: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)

    # -------------------- Properties (no duplicate metrics) --------------------

    @property
    def start_date(self) -> Optional[pd.Timestamp]:
        """First timestamp of the equity curve (or None if empty)."""
        return None if self.equity is None or self.equity.empty else self.equity.index[0]

    @property
    def end_date(self) -> Optional[pd.Timestamp]:
        """Last timestamp of the equity curve (or None if empty)."""
        return None if self.equity is None or self.equity.empty else self.equity.index[-1]

    @property
    def n_rebalances(self) -> int:
        """Number of recorded rebalances."""
        return 0 if self.rebal_dates is None else len(self.rebal_dates)

    @property
    def avg_turnover(self) -> Optional[float]:
        """Average one-way turnover per rebalance (mean of `turnover`)."""
        if self.turnover is None or self.turnover.empty:
            return None
        return float(self.turnover.mean())

    @property
    def ann_turnover(self) -> Optional[float]:
        """
        Annualized turnover assuming monthly cadence (12 * avg_turnover).
        If cadence differs, compute externally or pass cadence via `extras`.
        """
        if self.avg_turnover is None:
            return None
        return 12.0 * self.avg_turnover

    # -------------------- Lightweight summaries & export --------------------

    def to_summary_dict(self) -> dict:
        """
        JSON-friendly summary (no performance stats).
        Use your metrics module for CAGR/Sharpe/etc.
        """
        return {
            "name": self.name,
            "start_date": None if self.start_date is None else self.start_date.strftime("%Y-%m-%d"),
            "end_date": None if self.end_date is None else self.end_date.strftime("%Y-%m-%d"),
            "n_points": None if self.equity is None else int(len(self.equity)),
            "avg_turnover": self.avg_turnover,
            "ann_turnover": self.ann_turnover,
            "n_rebalances": self.n_rebalances,
            "entry_cost_frac": float(self.entry_cost_frac),
            "params": self.params,
        }

    def to_csvs(self, output_dir: str | Path, write_summary: bool = True) -> Dict[str, Path]:
        """
        Persist common artifacts for audits and plots.
        Writes: equity.csv, (optionally) turnover.csv, weights.csv, selections.json, summary.json.
        Returns path Dict
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        outpaths = {}

        # Equity
        eq = self.equity.copy()
        if eq.name is None:
            eq.name = "equity"
        this_out = out / "equity.csv"
        eq.to_csv(this_out, header=True)
        outpaths["equity"] = this_out

        # Turnover
        if self.turnover is not None:
            to = self.turnover.copy()
            if to.name is None:
                to.name = "turnover"
            this_out = out / "turnover.csv"
            to.to_csv(this_out, header=True)
            outpaths["turnover"] = this_out

        # Weights
        if self.weights is not None:
            this_out = out / "weights.csv"
            self.weights.to_csv(this_out)
            outpaths["weights"] = this_out

        # Selections (Timestamp keys → ISO strings)
        if self.selections is not None:
            sel_json = {pd.Timestamp(k).strftime("%Y-%m-%d"): v for k, v in self.selections.items()}
            this_out = out / "selections.json"
            (this_out).write_text(json.dumps(sel_json, indent=2))
            outpaths["selections"] = out / "selections.json"

        # Summary
        if write_summary:
            this_out = out / "summary.json"
            (this_out).write_text(json.dumps(self.to_summary_dict(), indent=2))
            outpaths["summary"] = this_out

        return outpaths

    # -------------------- Small utilities (no metrics logic) --------------------

    def rebase(self, value: float = 1.0) -> "StrategyResult":
        """Return a shallow copy with equity rebased to `value` at the first index point."""
        if self.equity is None or self.equity.empty:
            return self
        base = float(self.equity.iloc[0])
        if base == 0.0:
            return self
        scaled = self.equity * (value / base)
        return StrategyResult(
            name=self.name,
            equity=scaled,
            rebal_dates=self.rebal_dates,
            turnover=self.turnover,
            entry_cost_frac=self.entry_cost_frac,
            selections=self.selections,
            weights=self.weights,
            params=self.params.copy(),
            extras=self.extras.copy(),
        )

    def slice(
        self,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None
    ) -> "StrategyResult":
        """
        Return a shallow copy windowed to [start, end] on equity and aligned diagnostics.
        """
        if self.equity is None or self.equity.empty:
            return self

        eq = self.equity
        if start is not None:
            eq = eq[eq.index >= start]
        if end is not None:
            eq = eq[eq.index <= end]

        # Trim rebalances/turnover/weights/selections to the window
        new_rebals = self.rebal_dates
        if new_rebals is not None and len(new_rebals):
            if start is not None:
                new_rebals = new_rebals[new_rebals >= start]
            if end is not None:
                new_rebals = new_rebals[new_rebals <= end]

        new_turnover = None
        if self.turnover is not None:
            new_turnover = self.turnover
            if start is not None:
                new_turnover = new_turnover[new_turnover.index >= start]
            if end is not None:
                new_turnover = new_turnover[new_turnover.index <= end]

        new_weights = None
        if self.weights is not None:
            new_weights = self.weights
            if start is not None:
                new_weights = new_weights.loc[new_weights.index >= start]
            if end is not None:
                new_weights = new_weights.loc[new_weights.index <= end]

        new_selections = None
        if self.selections is not None:
            new_selections = {
                k: v for k, v in self.selections.items()
                if (start is None or k >= start) and (end is None or k <= end)
            }

        return StrategyResult(
            name=self.name,
            equity=eq,
            rebal_dates=new_rebals,
            turnover=new_turnover,
            entry_cost_frac=self.entry_cost_frac,
            selections=new_selections,
            weights=new_weights,
            params=self.params.copy(),
            extras=self.extras.copy(),
        )

    def validate(self) -> None:
        """Light checks for common pitfalls. Raises AssertionError on failure."""
        assert isinstance(self.equity, pd.Series), "equity must be a pd.Series"
        assert self.equity.index.is_monotonic_increasing, "equity index must be sorted ascending"
        if len(self.equity) > 0:
            assert np.isfinite(self.equity.iloc[0]), "equity first value must be finite"
        if self.turnover is not None:
            assert isinstance(self.turnover, pd.Series), "turnover must be a pd.Series or None"
            assert self.turnover.index.inferred_type in ("datetime64", "datetime"), "turnover index must be DatetimeIndex"
            assert (self.turnover >= 0).all(), "turnover must be non-negative"
        if self.rebal_dates is not None:
            assert isinstance(self.rebal_dates, pd.DatetimeIndex), "rebal_dates must be a DatetimeIndex"

    def compute_metrics(self, **kwargs):
        from mlbt.specs.metrics import compute_metrics        
        return compute_metrics(self, **kwargs)
