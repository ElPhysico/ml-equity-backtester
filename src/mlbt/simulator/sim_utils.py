# src/mlbt/simulator/sim_utils.py
"""
Contains utility functions for the simulator framework.
"""
import numpy as np
import string


def generate_random_tickers(
    rng: np.random.Generator,
    lengths: list[int],
    prefix: str = ""
) -> list[str]:
    """
    Generate unique random ticker symbols for the given list of lengths.
    """
    tickers = set()
    alphabet = np.array(list(string.ascii_uppercase))
    for L in lengths:
        while True:
            chars = rng.choice(alphabet, size=L)
            ticker = prefix + "_" + "".join(chars)
            if ticker not in tickers:
                tickers.add(ticker)
                break
    return list(tickers)


def generate_deterministic_tickers(
    N: int,
    prefix: str = "T",
    min_width: int = 3,
    start: int = 1
) -> list[str]:
    """
    Generate ticker labels like T001, T002, ..., with zero-padding sufficient for N.
    - min_width=3 gives 'T001' style; set to 1 if you want minimal padding.
    - start controls the starting index (default 1).
    """
    if N < 1:
        return []
    width = max(min_width, len(str(start + N - 1)))
    return [f"{prefix}_{i:0{width}d}" for i in range(start, start + N)]