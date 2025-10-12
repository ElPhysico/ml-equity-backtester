# src/mlbt/simulator/regimes.py
"""
This module holds pre-defined regimes and tools and helper for visualisation and other applications.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from collections.abc import Sequence




# ---------------- GBM ----------------

from mlbt.simulator.trajectories import simulate_gbm_trajectories

GBM_REGIMES = [
    {"label": "Bull - Low Vol",  "mu": 0.08, "sigma": 0.10},
    {"label": "Bull - High Vol", "mu": 0.08, "sigma": 0.30},
    {"label": "Sideways - Low Vol", "mu": 0.00, "sigma": 0.10},
    {"label": "Sideways - High Vol", "mu": 0.00, "sigma": 0.30},
    {"label": "Bear - Low Vol", "mu": -0.05, "sigma": 0.10},
    {"label": "Bear - High Vol", "mu": -0.05, "sigma": 0.30},
    {"label": "Crisis Shock", "mu": -0.20, "sigma": 0.45},
]

def get_GBM_regimes(
    print_table: bool = False
) -> dict[str, object]:
    df = pd.DataFrame(GBM_REGIMES)
    df["g = mu - 0.5*sigma^2"] = df["mu"] - 0.5 * df["sigma"]**2
    # df["TDY"] = TDY

    fmt = {
        "mu": "{:.2%}".format,
        "sigma": "{:.2%}".format,
        "g = mu - 0.5*sigma^2": "{:.2%}".format,
        "TDY": "{}".format,
    }
    df_disp = df[["label", "mu", "sigma", "g = mu - 0.5*sigma^2"]].rename(
        columns={"label": "Label", "mu": "μ", "sigma": "σ"}
    ).style.format(fmt)

    md = df[["label","mu","sigma","g = mu - 0.5*sigma^2"]].rename(
        columns={"label":"Label","mu":"μ","sigma":"σ"}
    ).to_string()

    if print_table:
        print(md)

    return GBM_REGIMES

def notebook_singular_GBM_graphs(
    regimes: dict[str, object] = GBM_REGIMES,
    calendar: pd.DatetimeIndex = pd.bdate_range("2000-01-01", "2009-12-31", freq="B"),
    n_tickers: int = 250,
    tdy: int = 260,
    seed: int = 1990
) -> None:
    cfg = {
    "calendar": calendar,
    "n_tickers": n_tickers,
    "tdy": tdy,
    "seed": seed
    }
    for reg in regimes:
        _ = visualize_gbm_regime(
            **cfg,
            regimes=[reg],
            label=reg["label"],
            y_lim=(0.04, 4),
            # savepath=f"../../docs/images/gbm_regimes/{reg['label'].replace("- ", "").replace(" ", "_")}.png"
        )

def notebook_mixed_GBM_graphs(
    regimes: dict[str, object] = [GBM_REGIMES[k] for k in (2, 3, 1, 6, 4, 0)],
    switch_at: Sequence[float] = [0.1, 0.2, 0.4, 0.48, 0.7],
    calendar: pd.DatetimeIndex = pd.bdate_range("2000-01-01", "2009-12-31", freq="B"),
    n_tickers: int = 250,
    tdy: int = 260,
    seed: int = 1990    
) -> None:
    cfg = {
    "calendar": calendar,
    "n_tickers": n_tickers,
    "tdy": tdy,
    "seed": seed
    }
    _ = visualize_gbm_regime(
        **cfg,
        regimes=regimes,
        switch_at=switch_at,
        label="Mixed GBM regimes"
    )


def cross_section_band(norm_px: pd.DataFrame, lo=0.10, hi=0.90) -> pd.DataFrame:
    """Cross-sectional percentile band at each date."""
    qlo = norm_px.quantile(lo, axis=1)
    qhi = norm_px.quantile(hi, axis=1)
    return pd.DataFrame({"lo": qlo, "hi": qhi})

def add_regime_shades(
    ax: plt.axes,
    segments: Sequence[dict[str, object]],
    *,
    alpha: float = 0.12,
    base_color: str = "tab:blue"
) -> list[Patch]:
    """
    Shade time segments on ax.

    segments: list of dicts like
      [{"start": "2000-01-01", "end": "2001-06-30", "label": "Sideways/Low"},
       {"start": "2001-07-01", "end": "2003-12-31", "label": "Bull/High"},
       ...]
    """
    handles = []
    # alternate light/darker tints for readability
    colors = [base_color, "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:cyan"]

    for i, seg in enumerate(segments):
        x0 = mdates.date2num(pd.to_datetime(seg["start"]))
        x1 = mdates.date2num(pd.to_datetime(seg["end"]))
        col = colors[i % len(colors)]
        ax.axvspan(x0, x1, facecolor=col, alpha=alpha, zorder=0, linewidth=0)
        # label at the top, centered in the span
        xm = x0 + (x1 - x0) * 0.5
        # y_shift = 0.125 if i%2==1 else 0
        y_shift = 0
        ax.text(mdates.num2date(xm), ax.get_ylim()[1]*(0.97 - y_shift), seg["label"],
                ha="center", va="top", fontsize=7, alpha=0.8)

        # legend handle (one per label)
        handles.append(Patch(facecolor=col, alpha=alpha, label=seg["label"]))

    # optional mini legend for regimes (separate from line legend)
    # ax.legend(handles=handles, title="Regimes", loc="upper right", frameon=True)
    return handles

def plot_gbm_regime(norm_px: pd.DataFrame,
                    segments: Sequence[dict[str, object]] | None = None,
                    title: str = "",
                    subtitle: str = "",
                    n_samples: int = 10,
                    alpha_paths: float = 0.35,
                    lw_paths: float = 1.0,
                    lw_index: float = 3.0,
                    lw_median: float = 2.2,
                    band_alpha: float = 0.12,
                    y_lim: tuple[float, float] = (0.01, 10.0),
                    savepath: str| Path | None = None
) -> tuple[Figure, plt.Axes]:
    """
    Plot normalized GBM trajectories with an equal-weight index and x-sec band.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # sample a few tickers
    cols = norm_px.columns.tolist()
    if len(cols) > n_samples:
        rng = np.random.default_rng(42)
        cols = list(rng.choice(cols, size=n_samples, replace=False))

    # paths
    ax.plot(norm_px[cols], linewidth=lw_paths, alpha=alpha_paths)

    # index
    label_mean   = "EW mean (arithmetic)"
    idx = norm_px.mean(axis=1)
    ax.plot(idx.index, idx.values, linewidth=lw_index, label=label_mean, color="black")

    # median
    label_median = "EW median (typical)"
    median = norm_px.median(axis=1)
    ax.plot(median.index, median.values, linewidth=lw_median, label=label_median, color="navy", linestyle="dashed")

    # band
    band = cross_section_band(norm_px, 0.10, 0.90)
    ax.fill_between(norm_px.index, band["lo"], band["hi"], alpha=band_alpha, label="10-90% band")

    # regimes
    if segments is not None:
        _ = add_regime_shades(ax, segments, alpha=0.07)

    # formatting
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)  # main title
    if subtitle:
        ax.set_title(subtitle, fontsize=10, loc="left", pad=10)  # subtitle
    caption = f"Showing {len(cols)} out of {norm_px.shape[1]} tickers for visual clarity.\nMean follows arithmetic expectation; median reflects typical (geometric) growth."
    fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=9, alpha=0.7)

    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized price (t=0 = 1.0)")
    ax.set_yscale("log")


    ax.set_xlim(norm_px.index[0], norm_px.index[-1])
    ax.margins(x=0) 
    ax.set_ylim(y_lim)
    from matplotlib.ticker import FuncFormatter
    scalar_fmt = FuncFormatter(lambda y, _: f"{y:g}")
    ax.yaxis.set_major_formatter(scalar_fmt)
    ax.yaxis.set_minor_formatter(scalar_fmt)

    ax.legend(loc="best", frameon=True)
    fig.tight_layout(rect=[0, 0.03, 1, 1.0])

    if savepath:
        fig.savefig(savepath, dpi=160, bbox_inches="tight")
    return fig, ax


def visualize_gbm_regime(
    calendar: pd.DatetimeIndex,
    regimes: Sequence[dict[str, object]],
    switch_at: Sequence[float] | None = None,
    n_tickers: int | None = None,
    tdy: int | None = None,
    seed: int | None = None,
    label: str | None = None,
    y_lim: tuple[float, float] | None = None,
    savepath: str | Path | None = None
) -> tuple[Figure, plt.axes]:
    allowed = {"n_tickers", "tdy", "seed"}
    sim_params = {k: v for k, v in locals().items() if k in allowed and v is not None}

    if switch_at is not None:
        if len(switch_at) != len(regimes) - 1:
            raise ValueError("There must be one more regime than switch dates!")
    elif switch_at is None and len(regimes) > 1:
        raise ValueError("There must be one more regime than switch dates!")
    
    # no safeguards for switch_at at this point
    switch_at = [calendar[int(len(calendar) * s)] for s in switch_at] if switch_at is not None else []
    switch_dates = [calendar[0]] + switch_at + [calendar[-1]]

    pxs = []
    metas = []
    reg_names = [r["label"] for r in regimes]

    for i, reg in enumerate(regimes):
        px, meta = simulate_gbm_trajectories(
            calendar=calendar[(calendar >= switch_dates[i]) & (calendar <= switch_dates[i+1])],
            **sim_params,
            ann_mu=reg["mu"],
            ann_sigma=reg["sigma"]
        )

        # normalize to previous price
        if pxs:
            px.columns = pxs[-1].columns
            px = px / px.iloc[0] * pxs[-1].iloc[-1]
            px = px.iloc[1:]
        pxs.append(px)
        metas.append(meta)
    
    px = pd.concat(pxs)
    norm = px / px.iloc[0]

    # subtitle creation
    if len(regimes) > 10:
        subtitle = f"N={norm.shape[1]} | TDY={tdy} | Regimes={len(regimes)}"
    elif len(regimes) > 3:
        subtitle = f"N={norm.shape[1]} | TDY={tdy} | Regimes={len(regimes)}\n"
        subtitle += r" $\rightarrow$ ".join(reg_names)
    else:
        params = metas[0]["params"]
        g = params["ann_mu"] - 0.5 * params["ann_sigma"]**2
        subtitle = f"{reg_names[0]}: μ={params["ann_mu"]:.2%} | σ={params["ann_sigma"]:.2%} | g={g:.2%}"

        for i, meta in enumerate(metas[1:]):
            params = meta["params"]
            g = params["ann_mu"] - 0.5 * params["ann_sigma"]**2
            subtitle += f"\n{reg_names[i+1]} μ={params["ann_mu"]:.2%} | σ={params["ann_sigma"]:.2%} | g={g:.2%}"
        subtitle += f"\nN={norm.shape[1]} tickers | TDY={tdy}"

    # segments for shading
    segments = []
    if len(regimes) >= 2:
        segments = [{
            "start": switch_dates[i],
            "end": switch_dates[i+1],
            "label": reg_names[i].replace(" - ", "\n")
        }
        for i in range(len(regimes))]


    return plot_gbm_regime(norm, segments=segments, title=label, subtitle=subtitle, y_lim=y_lim, savepath=savepath)