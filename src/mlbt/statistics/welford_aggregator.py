# src/mlbt/statistics/welford_aggregator.py
"""
Contains an aggregator dataclass that employs the Welford-online algorithm to aggregate mean and  for numpy arrays.
"""
import numpy as np
from scipy.stats import t
from dataclasses import dataclass, field


@dataclass
class Welford:
    """Implements Welford-online algorithm for numpy arrays."""
    n: int = 0
    mean: np.ndarray | None = field(default=None, repr=False)
    M2: np.ndarray | None = field(default=None, repr=False)

    @property
    def variance(self) -> np.ndarray | None:
        return self.M2 / (self.n - 1) if self.n > 1 else None

    @property
    def std(self) -> np.ndarray | None:
        v = self.variance
        return v ** 0.5 if v is not None else None

    @property
    def sem(self) -> np.ndarray | None:
        """Standard error of the mean."""
        v = self.variance
        return (v / self.n) ** 0.5 if (v is not None and self.n > 0) else None
    
    @property
    def ci95_mean(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        95% two-sided CI for the mean.
        """
        return self.ci(level=0.95, scale="sem")
    
    @property
    def ci95_typical(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        95% two-sided CI for the typical path.
        """
        return self.ci(level=0.95, scale="std")


    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)

        if self.n == 0:
            self.mean = np.zeros_like(x, dtype=float)
            self.M2 = np.zeros_like(x, dtype=float)

        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)
        

    def ci(self, level: float = 0.95, scale: str = "sem") -> tuple[np.ndarray, np.ndarray] | None:
        """
        Confidence interval for either the ensemble mean (scale='sem')
        or a typical path (scale='std'), for a given confidence level.
        """
        if self.n <= 1 or self.mean is None:
            return None
        df = self.n - 1

        if scale == "sem":
            sc = self.sem
        elif scale == "std":
            sc = self.std
        else:
            raise ValueError("'scale' must be 'sem' or 'std'")

        if sc is None:
            return None

        p = 1 - (1 - level) / 2
        c = t.ppf(p, df)

        lo = self.mean - c * sc
        hi = self.mean + c * sc
        return lo, hi
