# src/mlbt/cli_outputs.py
"""
Module to hold helper functions that format output nicely to the command line.
"""
import pandas as pd
import numpy as np


def cli_ann_log_growth_stats(
    stats: pd.DataFrame,
    T: float,
    alt: str = "less"
) -> str:
    if alt == "less":
        c = "<"
    elif alt == "greater":
        c = ">"
    r_string = []
    for r in stats.itertuples(index=False):
        lines = []
        title = f"[{r.strategy}] vs [{r.benchmark}]"
        lines += [title, "-" * len(title)]
        lines += ["| delta log growth-rate:",
                  f"| - d = {r.d:>8.4f} -- 95% CI: [{r.d_lo:>8.4f}, {r.d_hi:>8.4f}]",
                #   f"| - 95% CI: [{r.d_lo:.4f}, {r.d_hi:.4f}]",
                  "|",
                  "| terminal wealth ratio:",
                  f"| - V = {np.exp(r.d * T):>8.4f} -- 95% CI [{np.exp(r.d_lo * T):>8.4f}, {np.exp(r.d_hi * T):>8.4f}]",
                #   f"| - 95% CI [{np.exp(r.d_lo * T):.4f}, {np.exp(r.d_hi * T):.4f}]",
                  "|",
                  "| win-rate:",
                  f"| - r = {r.win_rate:>8.4f} -- 95% CI [{r.win_lo:>8.4f}, {r.win_hi:>8.4f}]",
                #   f"| - 95% CI [{r.win_lo:.4f}, {r.win_hi:.4f}]",
                  "|",
                  f"| p-value (H₀: r=0.5, H₁: r{c}0.5):",
                  f"| - p = {r.p_value:>8.2e}"]
        if r.p_value < 0.05:
            lines += ["|   ↳ Reject H₀ at 5% level"]
        else:
            lines += ["|   ↳ Fail to reject H₀"]

        r_string.append("\n".join(lines))
    return "\n\n".join(r_string)



def md_ann_log_growth_stats(
    stats: pd.DataFrame,
    T: float,
    alt: str = "less"
) -> str:
    if alt == "less":
        c = "<"
    elif alt == "greater":
        c = ">"
    sections = []
    for r in stats.itertuples(index=False):
        header = f"### {r.strategy} vs {r.benchmark}"
        bullets = [
            f"**Δ log growth-rate:** `d = {r.d:.4f}`  \n95% CI: [`{r.d_lo:.4f}`, `{r.d_hi:.4f}`]",
            f"**Terminal wealth ratio:** `V = {np.exp(r.d * T):.4f}`  \n95% CI: [`{np.exp(r.d_lo * T):.4f}`, `{np.exp(r.d_hi * T):.4f}`]",
            f"**Win-rate:** `r = {r.win_rate:.4f}`  \n95% CI: [`{r.win_lo:.4f}`, `{r.win_hi:.4f}`]",
            f"**p-value (H₀: r=0.5, H₁: r{c}0.5):** `p = {r.p_value:.2e}`",
        ]
        if r.p_value < 0.05:
            bullets.append("↳ ✅ *Reject H₀ at 5% level*")
        else:
            bullets.append("↳ ❌ *Fail to reject H₀*")
        sections.append(header + "\n\n" + "\n".join(f"- {b}" for b in bullets))
    return "\n\n".join(sections)

def md_ann_log_growth_stats_table(
    stats: pd.DataFrame,
    T: float,
    alt: str = "less"
) -> str:
    if alt == "less":
        c = "<"
    elif alt == "greater":
        c = ">"
    sections = []
    for r in stats.itertuples(index=False):
        header = f"### {r.strategy} vs {r.benchmark}"
        table = [
            "| Metric | Estimate | 95% CI |",
            "|:-------|---------:|:-------|",
            f"| Δ log growth-rate | `{r.d:.4f}` | [`{r.d_lo:.4f}`, `{r.d_hi:.4f}`] |",
            f"| Terminal wealth ratio | `{np.exp(r.d * T):.4f}` | [`{np.exp(r.d_lo * T):.4f}`, `{np.exp(r.d_hi * T):.4f}`] |",
            f"| Win-rate | `{r.win_rate:.4f}` | [`{r.win_lo:.4f}`, `{r.win_hi:.4f}`] |",
            f"| p-value (H₀: r=0.5, H₁: r{c}0.5) | `{r.p_value:.2e}` |  |",
        ]
        if r.p_value < 0.05:
            decision = "✅ *Reject H₀ at 5% level*"
        else:
            decision = "❌ *Fail to reject H₀*"
        sections.append(header + "\n\n" + "\n".join(table) + "\n\n" + decision)
    return "\n\n---\n\n".join(sections)


def prep_metrics_df(
    metrics_df: pd.DataFrame,
    delta_log_growth_stats: pd.DataFrame
) -> pd.DataFrame:
    col_mapping = {
        "total_return": "Total Return",
        "cagr": "CAGR",
        "arithmetic_mean": "Arithmetic Mean",
        "vol_ann": "Vol",
        "sharpe": "Sharpe",
        "max_drawdown": "MaxDD",
        "ann_turnover": "Turnover",
    }

    df = metrics_df.rename(columns=col_mapping)[list(col_mapping.values())]
    tmp = delta_log_growth_stats.query('benchmark == "Monthly Rebalance EW"')[["strategy", "d", "win_rate"]]
    df[["Delta log growth vs MR", "Win-rate vs MR"]] = (
        tmp.set_index("strategy")[["d", "win_rate"]]
        .reindex(df.index)   # align to df's order
    )
    return df.sort_values("Sharpe", ascending=False).round(4).to_markdown(index=True)