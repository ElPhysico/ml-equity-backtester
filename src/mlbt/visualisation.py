# src/mlbt/visualisation.py
"""
Visualisation tools and helpers.

This module holds tools and helpers to visualise a variety of data, such as equity curves, universes, and statistics.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections.abc import Sequence, Callable

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import seaborn as sns

from mlbt.specs.strategy_spec import StrategyResult
from mlbt.specs.universe_spec import UniverseSpec


# helpers ######################################################################

def _cross_section_band(norm_px: pd.DataFrame, lo=0.10, hi=0.90) -> pd.DataFrame:
    """Cross-sectional percentile band at each date."""
    qlo = norm_px.quantile(lo, axis=1)
    qhi = norm_px.quantile(hi, axis=1)
    return pd.DataFrame({"lo": qlo, "hi": qhi})

def _add_regime_shades(
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
    colors = [base_color, "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:cyan"]

    for i, seg in enumerate(segments):
        x0 = mdates.date2num(pd.to_datetime(seg["start"]))
        x1 = mdates.date2num(pd.to_datetime(seg["end"]))
        col = colors[i % len(colors)]
        ax.axvspan(x0, x1, facecolor=col, alpha=alpha, zorder=0, linewidth=0)
        xm = x0 + (x1 - x0) * 0.5
        y_shift = 0
        ax.text(mdates.num2date(xm), ax.get_ylim()[1]*(0.97 - y_shift), seg["label"],
                ha="center", va="top", fontsize=8, alpha=0.8)

        handles.append(Patch(facecolor=col, alpha=alpha, label=seg["label"]))
    return handles


# Plotting equity curves #######################################################

def plot_equities(
    equities: Sequence[StrategyResult] | Sequence[pd.Series] | StrategyResult | pd.Series,
    names: Sequence[str] | None = None,
    bands: tuple[Sequence[pd.Series], Sequence[pd.Series]] | None = None,
    title: str | None = None,
    log_y: bool = False,
    save: bool = False,
    out_dir: Path | None = None,
    out_name: str = "equities.png"
) -> Figure:
    # check for input and convert to list
    if isinstance(equities, (pd.Series, StrategyResult)):
        equities = [equities]
    elif not isinstance(equities, Sequence):
        raise TypeError("Input must be a StrategyResult, pd.Series, or a sequence thereof")
    
    # normalize
    norm_equities = []
    norm_names = []
    # print("this is equities:", equities)
    for eq in equities:
        if isinstance(eq, StrategyResult):
            norm_equities.append(eq.equity)
        elif isinstance(eq, pd.Series):
            norm_equities.append(eq)
        else:
            raise TypeError(f"Unsupported element type: {type(eq)}")
        norm_names.append(eq.name)

    if names is None:
        names = norm_names
    df = pd.concat(norm_equities, axis=1, keys=names).dropna(how="all")
    
    sns.set_theme(context="talk", style="whitegrid")
    fig = plt.figure(figsize=(12, 6))

    # curves
    ax = sns.lineplot(data=df)

    # bands
    if bands is not None:
        palette = sns.color_palette(n_colors=len(df.columns))
        lower = bands[0]
        upper = bands[1]
        for i, name in enumerate(df.columns):
            t = df.index
            plt.fill_between(
                t,
                lower[name],
                upper[name],
                alpha=0.12,
                color=palette[i]
                # label=f"{name} 95% band"
            )

    if title is None:
        title = "Normalised Equity Curves"
    plt.title(title)
    if log_y:
        ax.set_yscale("log")
        scalar_fmt = FuncFormatter(lambda y, _: f"{y:g}")
        ax.yaxis.set_major_formatter(scalar_fmt)
        ax.yaxis.set_minor_formatter(scalar_fmt)
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
    return fig


# Plotting universes ###########################################################

def plot_universe(
    px: pd.DataFrame,
    univ: UniverseSpec | None = None,
    title: str = "",
    subtitle: str = "",
    n_samples: int = 10,
    alpha_paths: float = 0.3,
    lw_paths: float = 1.0,
    lw_index: float = 2.2,
    lw_median: float = 1.5,
    band_alpha: float = 0.12,
    # y_lim: tuple[float, float] | None = None,
    savepath: str| Path | None = None
) -> Figure:
    """
    Plot 'n_sample' normalized universe trajectories with an equal-weight universe index, median, and 10-90 band.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # sample a few tickers
    cols = px.columns.tolist()
    if len(cols) > n_samples:
        rng = np.random.default_rng(42)
        cols = list(rng.choice(cols, size=n_samples, replace=False))

    # ensure normalisation
    norm_px = px / px.iloc[0]

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
    band = _cross_section_band(norm_px, 0.10, 0.90)
    ax.fill_between(norm_px.index, band["lo"], band["hi"], alpha=band_alpha, label="10-90% band")

    # segments
    if univ is not None:
        if univ.meta is not None and "segments" in univ.meta["params"]:
            if len(univ.meta["params"]["segments"]) > 1:
                _ = _add_regime_shades(ax, univ.meta["params"]["segments"])

    # titles and caption
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    if subtitle:
        ax.set_title(subtitle, fontsize=10, loc="left", pad=10)
    caption = f"Showing {len(cols)} out of {norm_px.shape[1]} tickers for visual clarity.\nMean follows arithmetic expectation; median reflects typical (geometric) growth."
    fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=9, alpha=0.7)

    # formatting
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized price")
    ax.set_yscale("log")
    ax.set_xlim(norm_px.index[0], norm_px.index[-1])
    ax.margins(x=0)
    scalar_fmt = FuncFormatter(lambda y, _: f"{y:g}")
    ax.yaxis.set_major_formatter(scalar_fmt)
    ax.yaxis.set_minor_formatter(scalar_fmt)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout(rect=[0, 0.03, 1, 1.0])

    if savepath:
        fig.savefig(savepath, dpi=160, bbox_inches="tight")
    return fig, ax


# Plotting statistics ##########################################################

def plot_ann_log_growth_statistics(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    x: str = "strategy",
    y: str = "ann_log_growth",
    hue: str = "benchmark"
) -> tuple[Figure, Axes]:
    fig = plt.figure(figsize=(12, 6))
    # if n_seeds >= 20:
    ax = sns.violinplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        split=False,
        inner="quartile",
        gap=0,
        cut=0
    )

    # find violin bodies for arrangements
    violin_bodies = [
        p for p in ax.collections
        if isinstance(p, mpl.collections.PolyCollection)
        and p.get_facecolor().shape[0] > 0
    ]
    pos_map = {}
    idx = 0
    for s in df[x].unique():
        pos_map[s] = {}
        for b in df[hue].unique():
            verts = violin_bodies[idx].get_paths()[0].vertices
            pos_map[s][b] =  np.mean(verts[:, 0])
            idx += 1

    # # plot mean and CI
    for s in df[x].unique():
        for b in df[hue].unique():
            x = pos_map[s][b]
            mask = (
                (stats["strategy"] == s) &
                (stats["benchmark"] == b)
            )
            row = stats.loc[mask]
            ax.scatter(x, row.d, color="white", edgecolor="black", zorder=5)
            ax.vlines(x, row.d_lo, row.d_hi, color="white", linewidth=2, zorder=4)
    
    # legend formatting
    sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, 1.1), ncol=len(df[hue].unique()), frameon=False, title=None)

    # y axis to bps/year
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x*1e4:.0f}"))
    ax.set_ylabel("annual log growth (bps/year)")

    # zero line plus no xlabel
    ax.grid(False)
    ax.axhline(0, color="lightgrey", linewidth=1, linestyle="dashed")
    ax.set_xlabel("")

    plt.tight_layout(rect=[0, 0.0, 1, 1.0])

    # plt.show()

    return fig, ax


def plot_vol_curve(
    mean_df: pd.DataFrame,
    bench: str
) -> tuple[plt.Figure, plt.Axes]:
    x = "n_tickers"
    y = "vol_ann"
    y_lo = "vol_ann_low"
    y_hi = "vol_ann_high"
    df_sorted = mean_df[[x, y, y_lo, y_hi]].dropna().sort_values(by=x).drop(bench)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        df_sorted[x],
        df_sorted[y],
        yerr=[df_sorted[y] - df_sorted[y_lo], df_sorted[y_hi] - df_sorted[y]],
        fmt="o",
        ecolor="black",
        elinewidth=1,
        capsize=3,
        alpha=0.8,
        # label="Sample mean vol for Top-N strategy"
        label=r"$\sigma_\mathrm{Top-N}$"
    )    

    # plot theoretical scaling based on bench
    mapping = {"Monthly Rebalance EW": "MR", "Buy & Hold EW": "BH"}
    x_min, x_max = df_sorted[x].min(), df_sorted[x].max()
    x_support = np.linspace(x_min, x_max, 1000)
    for b in bench:
        guide_vol = mean_df.loc[b, "vol_ann"]
        K = mean_df.loc[b, "n_tickers"]
        y = guide_vol * np.sqrt(K) / np.sqrt(x_support)
        ax.plot(x_support, y, linestyle="--", label=fr"$\sigma_\mathrm{{{mapping[b]}}} \sqrt{{K/N}}$")


    # guide_vol = mean_df.loc[bench, "vol_ann"] #* np.sqrt( 1/x )
    # rho_bar = 7.426933439967973e-06
    # K = mean_df.loc[bench, "n_tickers"]
    # print(guide_vol, "theoretical guide vol:", 0.25/np.sqrt(K))
    # num = (1.0 / x_support) + ((x_support - 1.0) / x_support) * rho_bar
    # den = (1.0 / K) + ((K - 1.0) / K) * rho_bar
    # y_support_adj = guide_vol * np.sqrt(num/den)
    # y_theory = 0.25/np.sqrt(K) * np.sqrt(num/den)

    # y_support = guide_vol * np.sqrt(K) / np.sqrt(x_support)
    # ax.plot(x_support, y_support, color="red", linestyle="--", label="Theoretical ratio")
    # ax.plot(x_support, y_support_adj, color="navy", linestyle="--", label="Theoretical ratio correlation adjusted", alpha=0.6)
    # ax.plot(x_support, y_theory, color="purple", linestyle="--", label="Theoretical ratio correlation adjusted", alpha=0.6)

    # print(df_sorted[y].values - guide_vol * np.sqrt(K) / np.sqrt(df_sorted[x].values))

    fig.suptitle("Volatility scaling with Top-N")
    ax.set_xlabel("N")
    ax.set_ylabel("Annualized volatility")
    ax.legend()
    return fig, ax


    

# Plotting distributions #######################################################

def plot_gbm_distribution(
    t: float,
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
) -> tuple[Figure, tuple[Axes, Axes]]:
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