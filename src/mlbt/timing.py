# src/mlbt/timing.py
"""
Module for timing performance.
"""
# timing.py
from __future__ import annotations
from time import perf_counter
from contextlib import contextmanager
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Optional

class SectionTimer:
    """
    Lightweight timing engine to measure named code sections inside a function.
    Usage:
        timer = SectionTimer()
        with timer.section("build prefix logs"):
            ...
        with timer.section("block loop"):
            ...
        # print summary
        timer.report()
        # or get raw data
        stats = timer.summary()
    """
    def __init__(self, sink: Optional[Callable[[str], None]] = print):
        self._records: List[Tuple[str, float]] = []
        self._sink = sink

    @contextmanager
    def section(self, name: str):
        t0 = perf_counter()
        try:
            yield
        finally:
            dt = perf_counter() - t0
            self._records.append((name, dt))

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Aggregate timings per section: total, count, avg."""
        agg = defaultdict(lambda: {"total": 0.0, "count": 0, "avg": 0.0})
        for name, dt in self._records:
            a = agg[name]
            a["total"] += dt
            a["count"] += 1
        for a in agg.values():
            a["avg"] = a["total"] / a["count"]
        return dict(agg)

    def report(self, title: str = "Timing summary"):
        if self._sink is None:
            return
        data = self.summary()
        self._sink(f"\n{title}")
        self._sink("-" * len(title))
        for name, stats in sorted(data.items(), key=lambda kv: kv[1]["total"], reverse=True):
            self._sink(f"{name:30s}  total={stats['total']:.6f}s  "
                       f"count={stats['count']}  avg={stats['avg']:.6f}s")

def time_call(fn: Callable) -> Callable:
    """Optional: decorator to time a whole function call."""
    def wrapper(*args, **kwargs):
        t0 = perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            print(f"{fn.__name__} took {perf_counter() - t0:.6f}s")
    return wrapper