#!/usr/bin/env python3
"""
benchmarks.py — Run buy-and-hold strategy on benchmarks and equal-weight portfolio of all stocks.
"""


import argparse
from pathlib import Path
import logging
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from mlbt.log_utils import setup_logging
from mlbt.yaml_io import read_yaml
from mlbt.load_prices import load_prices, flag_suspect_splits
from mlbt.strategies.buy_and_hold import buy_and_hold
from mlbt.strategies.monthly_rebalance import monthly_rebalance


# ---------------- CLI ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run buy-and-hold strategy on benchmarks and equal-weight portfolio of all stocks.")
    p.add_argument("--in-dir", type=str, default="data/equity_data", help="Partitioned Parquet root with OHLCV.")
    p.add_argument("--out-dir", type=str, default="results/benchmarks", help="Root for benchmark results.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

# ---------------- Helpers ----------------




# ---------------- Main ----------------

def main():
    args = parse_args()
    setup_logging(verbose=args.verbose)
    
    config = read_yaml("config/run2.yaml")
    benchmarks = config.get("benchmarks", [])
    stocks = config.get("stocks", [])
    start_date = config.get("min_date", None)
    end_date = config.get("max_date", None)

    px = load_prices(in_dir=args.in_dir,
                     tickers=benchmarks+stocks,
                     start=start_date,
                     end=end_date)
    
    flagged = flag_suspect_splits(px)
    if not flagged.empty:
        logging.warning("Suspicious price moves detected (potential splits or bad data). Please investigate.")

    cost_bps=5.0
    bench_equities = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for bench in benchmarks:
        res = buy_and_hold(px[[bench]], cost_bps=cost_bps)
        bench_equities.append(res.equity)
        out = out_dir / f"{bench}_buy_and_hold.csv"
        res.equity.to_csv(out, header=True)
        logging.info(f"Wrote [{bench}] benchmark equity to {out}")

    stock_res = buy_and_hold(px[stocks], cost_bps=cost_bps)
    out = out_dir / "all_stocks_equal_weight_buy_and_hold.csv"
    stock_res.equity.to_csv(out, header=True)
    logging.info(f"Wrote all-stocks ({len(stocks)}) equal weight buy-and-hold equity to {out}")

    m_rebal_res = monthly_rebalance(px[stocks], cost_bps=cost_bps)
    out = out_dir / "all_stocks_monthly_rebalance.csv"
    m_rebal_res.equity.to_csv(out, header=True)
    logging.info(f"Wrote all-stocks ({len(stocks)}) monthly-rebalance equity to {out}")


    # plotting
    curves = {bench: s for bench, s in zip(benchmarks, bench_equities)}
    curves[f"Stocks ({len(stocks)}) (Equal-Weight B&H)"] = stock_res.equity
    curves[f"Stocks ({len(stocks)}) (Equal-Weight Monthly-Rebalance)"] = m_rebal_res.equity

    df_plot = pd.concat(curves, axis=1)  # columns = strategy names
    df_plot = df_plot.dropna(how="all")  # safety

    plot_df = (
        df_plot
        .reset_index(names="date")
        .melt(id_vars="date", var_name="strategy", value_name="equity")
        .dropna(subset=["equity"])
    )

    plt.figure(figsize=(10, 5))
    ax = sns.lineplot(data=plot_df, x="date", y="equity", hue="strategy")
    # ax.set_yscale("log")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.set_title("Buy & Hold (FX-agnostic, index mode)")

    plt.tight_layout()
    fig_path = out_dir / "benchmarks_equity.png"
    # plt.show()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logging.info(f"Wrote plot to {fig_path}")



if __name__ == "__main__":
    main()