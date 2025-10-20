# src/mlbt/simulator/trajectories.py
"""
Module to create different kinds of equity trajectories.
"""
import numpy as np
import pandas as pd
import logging

from typing import Any

from mlbt.simulator.meta import build_trajectories_meta
from mlbt.calendar import tdy_from_index
from mlbt.specs.universe_spec import UniverseSpec
from mlbt.simulator.sim_utils import generate_deterministic_tickers, generate_random_tickers


# ---------------- Simulator ----------------

def simulate_universe(        
    univ: UniverseSpec,
    calendar: pd.DatetimeIndex | None = None,
    start: str | None = None,
    end: str | None = None,
    rng: np.random.Generator | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if calendar is not None:
        cal = calendar
    elif start is not None and end is not None:
        cal = pd.bdate_range(start, end, freq="B")
    else:
        raise ValueError("Requires a 'pd.DateTimeIndex' or start and end strings in 'YYYY-MM-DD' format.")

    simulator = univ.simulator
    px, meta = simulator(
        calendar=cal,
        n_tickers=univ.n_tickers,
        **univ.sim_params,
        ticker_prefix=univ.key,
        rng=rng
    )

    univ.meta = meta

    return px, meta


# ---------------- GBM ----------------

def simulate_gbm_trajectories(
    calendar: pd.DatetimeIndex,
    *,
    n_tickers: int,
    mu: list[float] | float,    # annualized
    sigma: list[float] | float, # annualized
    switch_at: list[float] | None = None,
    labels: list[str] | None = None,
    random_ticker_labels: bool = False,
    ticker_prefix: str = "",
    rng: np.random.Generator | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if rng is None:
        rng = np.random.default_rng(1990)

    # retrieve trading days per year from calendar
    tdy = tdy_from_index(calendar)
    if calendar.freqstr not in ["D", "B"]:
        logging.warning(F"Calendar's period ({calendar.freqstr}) is not daily ('D' or 'B')")

    # convert single values to list
    if not isinstance(mu, list):
        mu = [mu]
    if not isinstance(sigma, list):
        sigma = [sigma]

    # some safeguards
    if switch_at is not None:
        if len(switch_at) != len(mu) - 1:
            raise ValueError("There must be one more regime than switch dates!")
    elif switch_at is None and len(mu) > 1:
        raise ValueError("There must be one more regime than switch dates!")
    
    # no safeguards for switch_at at this point
    switch_at = [calendar[int(len(calendar) * s)] for s in switch_at] if switch_at is not None else []
    switch_dates = [calendar[0]] + switch_at + [calendar[-1]]

    # create tickers
    if random_ticker_labels:
        tickers = generate_random_tickers(rng, rng.integers(low=4, high=5, size=n_tickers), prefix=ticker_prefix)
    else:
        tickers = generate_deterministic_tickers(n_tickers, prefix=ticker_prefix)

    # prepare simulations
    lnS_0 = np.log(rng.integers(low=1, high=1000, size=n_tickers, endpoint=True))[:, None]
    mu_step = [m / tdy for m in mu]
    sigma_step = [s / np.sqrt(tdy) for s in sigma]   

    # simulate
    pxs = []
    segments = []
    for i in range(len(mu_step)):
        cal = calendar[(calendar >= switch_dates[i]) & (calendar <= switch_dates[i+1])]
        segments.append({
            "start": cal[0].strftime("%Y-%m-%d"),
            "end": cal[-1].strftime("%Y-%m-%d"),
            "label": labels[i] if labels is not None else f"Segment {i+1}"
        })
        n = len(cal)
        Z = rng.standard_normal((n_tickers, n - 1))
        dln = (mu_step[i] - 0.5 * sigma_step[i]**2) + sigma_step[i] * Z
        lnS = lnS_0 + np.cumsum(dln, axis=1)

        S = np.exp(np.concatenate([lnS_0, lnS], axis=1))
        if pxs:
            S = S / S[:, [0]] * pxs[-1][:, [-1]]
            S = S[:, 1:]
        pxs.append(S)

    pxs = np.concatenate(pxs, axis=1)
    px_wide = pd.DataFrame(pxs.T, index=calendar, columns=tickers)

    ss = rng.bit_generator._seed_seq

    meta = build_trajectories_meta(
        n_tickers=n_tickers,
        calendar=calendar,
        params={
            "mu": mu,
            "sigma": sigma,
            "switch_dates": [s.strftime("%Y-%m-%d") for s in switch_dates],
            "segments": segments,
            "TDY": tdy,
        },
        rng={
            "seed_entropy": int(ss.entropy),
            "spawn_key": [int(k) for k in ss.spawn_key],
            "state": int(ss.generate_state(1)[0])
        }
    )
    
    return px_wide, meta
