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
        columns={"label":"Label","mu": "μ","sigma":"σ","g = mu - 0.5*sigma^2":"g = μ - 0.5*σ^2"}
    ).to_string()

    if print_table:
        print(md.replace("mu", r"$\mu$"))

    return GBM_REGIMES

def notebook_single_GBM_graph(
    mu: float = 0.05,
    sigma: float = 0.2,
    label: str = "Example GBM paths",
    calendar: pd.DatetimeIndex = pd.bdate_range("2000-01-01", "2009-12-31", freq="B"),
    n_tickers: int = 10,
    tdy: int = 260,
    seed: int = 1990
) -> None:
    cfg = {
    "calendar": calendar,
    "n_tickers": n_tickers,
    "tdy": tdy,
    "seed": seed
    }
    reg = {
        "label": label,
        "mu": mu,
        "sigma": sigma
    }
    _ = visualize_gbm_regime(
        **cfg,
        regimes=[reg],
        label=reg["label"],
        # savepath=f"../../docs/images/gbm_regimes/{reg['label'].replace("- ", "").replace(" ", "_")}.png"
    )

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
                ha="center", va="top", fontsize=8, alpha=0.8)

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
                    alpha_paths: float = 0.3,
                    lw_paths: float = 1.0,
                    lw_index: float = 2.2,
                    lw_median: float = 1.5,
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
        subtitle = f"N={norm.shape[1]} tickers | TDY={tdy} | Regimes={len(regimes)}"
    elif len(regimes) > 3:
        subtitle = f"N={norm.shape[1]} tickers | TDY={tdy} | Regimes={len(regimes)}\n"
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


def plot_gbm_distributions(t: float,
                           mu: float,
                           sigma: float,
                           S0: float = 1.0,
                           n_samples: int = 100_000,
                           bins_ln: int = 100,
                           bins_S: int = 120,
                           rng: np.random.Generator | None = None,
                           title: str = "GBM terminal distributions at fixed t",
                           subtitle: str = "",
                           savepath: str | Path | None = None
                           ) -> tuple[Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Plot the distributions of ln S(t) and S(t) side-by-side for GBM:
        S(t) = S0 * exp((mu - 0.5*sigma^2) * t + sigma * W_t),  W_t ~ N(0, t)

    Left:  ln S(t) ~ Normal(m, v) with m = ln S0 + (mu - 0.5*sigma^2)t, v = sigma^2 t
           Show histogram, theoretical normal PDF, and vertical lines at mean=median=mode=m.

    Right: S(t) ~ Lognormal(m, v) with same (m, v) as above
           Show histogram, theoretical lognormal PDF, and vertical lines at:
               mean   = S0 * exp(mu * t)
               median = S0 * exp((mu - 0.5*sigma^2) * t)
               mode   = S0 * exp((mu - sigma**2) * t)

    Parameters
    ----------
    t, mu, sigma : float
        GBM parameters at the terminal time t.
    S0 : float, default 1.0
        Initial price / normalization.
    n_samples : int, default 100_000
        Number of Monte Carlo samples to draw for the histograms.
    bins_ln : int, default 100
        Histogram bins for ln S.
    bins_S : int, default 120
        Histogram bins for S.
    rng : np.random.Generator | None
        Optional RNG; if None, uses np.random.default_rng(42).
    title, subtitle : str
        Figure titles.
    savepath : str | Path | None
        If given, save the figure.

    Returns
    -------
    fig, (ax_ln, ax_S)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Theoretical parameters for ln S
    m = np.log(S0) + (mu - 0.5 * sigma**2) * t
    v = (sigma**2) * t
    s = np.sqrt(v)

    # Monte Carlo samples
    Wt = rng.normal(loc=0.0, scale=np.sqrt(t), size=n_samples)
    lnS = m + sigma * Wt  # = ln S(t)
    S = np.exp(lnS)       # = S(t)

    # Theoretical markers in S-space
    S_mean   = S0 * np.exp(mu * t)
    S_median = S0 * np.exp((mu - 0.5 * sigma**2) * t)
    S_mode   = S0 * np.exp((mu - 1.5 * sigma**2) * t)

    # Create figure
    fig, (ax_ln, ax_S) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Left: ln S(t) (Normal) ----
    # Histogram (density)
    ax_ln.hist(lnS, bins=bins_ln, density=True, alpha=0.35, edgecolor='none', label="MC histogram")

    # Theoretical Normal PDF
    # pdf(x) = (1/(s*sqrt(2π))) * exp(-0.5*((x-m)/s)^2)
    x_ln = np.linspace(lnS.min(), lnS.max(), 800)
    pdf_ln = (1.0 / (s * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x_ln - m)/s)**2)
    ax_ln.plot(x_ln, pdf_ln, linewidth=2.0, label="Normal PDF", alpha=0.5, linestyle="dashed")

    # Mean/median/mode (all equal to m)
    ax_ln.axvline(m, linestyle="--", linewidth=1.5, label="mean=median=mode")

    ax_ln.set_title("Distribution of ln S(t)")
    ax_ln.set_xlabel("ln S")
    ax_ln.set_ylabel("Density")
    ax_ln.legend(loc="best", frameon=True)

    # ---- Right: S(t) (Lognormal) ----
    # Histogram (density); avoid plotting extreme tail bins too wide by clipping max
    # (Let numpy decide bins; we already pass bins_S)
    ax_S.hist(S, bins=bins_S, density=True, alpha=0.35, edgecolor='none', label="MC histogram")

    # Theoretical Lognormal PDF: for y>0, f(y) = (1/(y*s*sqrt(2π))) * exp(-0.5*((ln y - m)/s)^2)
    y_min = max(1e-12, S.min())
    y_max = np.quantile(S, 0.999) * 1.5  # extend a bit beyond 99.9% quantile for tail visibility
    y = np.linspace(y_min, y_max, 1000)
    pdf_S = (1.0 / (y * s * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((np.log(y) - m)/s)**2)
    ax_S.plot(y, pdf_S, linewidth=2.0, label="Lognormal PDF", alpha=0.5, linestyle="dashed")

    # Mark mean, median, mode
    ax_S.axvline(S_mean,   linestyle="-",  linewidth=1.5, label=f"mean = {S_mean:.3g}")
    ax_S.axvline(S_median, linestyle="--", linewidth=1.5, label=f"median = {S_median:.3g}")
    ax_S.axvline(S_mode,   linestyle=":",  linewidth=1.5, label=f"mode = {S_mode:.3g}")

    ax_S.set_title("Distribution of S(t)")
    ax_S.set_xlabel("S")
    ax_S.set_ylabel("Density")
    ax_S.set_xlim(left=0.0)  # S>=0
    ax_S.legend(loc="best", frameon=True)

    # ---- Figure titles / caption ----
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    if subtitle:
        ax_ln.set_title(ax_ln.get_title() + f"\n{subtitle}", fontsize=10)
        ax_S.set_title(ax_S.get_title() + f"\n{subtitle}", fontsize=10)

    caption = (
        "Left: ln S(t) ~ N(m, v) with m = ln S0 + (μ - ½σ²)t, v = σ²t (mean=median=mode).\n"
        "Right: S(t) is lognormal — right-skewed. Vertical lines show mean, median, and mode."
    )
    fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=9, alpha=0.7)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    if savepath:
        fig.savefig(savepath, dpi=160, bbox_inches="tight")

    return fig, (ax_ln, ax_S)
