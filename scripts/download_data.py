#!/usr/bin/env python3
"""
download_data.py - Download daily OHLCV data from Alpha Vantage API and write to Parquet.

Usage examples:
    # Bootstrap everything (20y) from universe file
    python scripts/download_data.py --mode init --universe-file config/universes/2025-09_default_universe.csv

    # Update (last 100d) everything from universe file
    python scripts/download_data.py --mode update --universe-file config/universes/2025-09_default_universe.csv

    # Manual ticker list
    python scripts/download_data.py --mode init --tickers AAPL MSFT GOOGL AMZN
"""

from pathlib import Path
import logging
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from alpha_vantage.timeseries import TimeSeries
import random
import time
from datetime import datetime, UTC
import json

from typing import Optional, List

from mlbt.log_utils import setup_logging
from mlbt.load_universe import load_universe
from mlbt.api_key import get_api_key


# ---------------- CLI ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download daily OHLCV data from Alpha Vantage API and write to Parquet.")
    p.add_argument("--mode", choices=["init", "update"], default="update",
                   help="init = full history; update = compact unless big gap (then full).")
    p.add_argument("--universe-file", type=str, default="config/universes/2025-09_default_universe.csv", help="CSV (with a 'symbol' column) or text file (one ticker per line).")
    p.add_argument("--tickers", nargs="*", default=None, help="Manual list of tickers (overrides universe-file).")
    p.add_argument("--out-dir", type=str, default="data/equity_data", help="Parquet dataset root.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ---------------- Helpers ----------------

def get_existing_max_date_for_ticker(out_dir: Path, ticker: str) -> Optional[pd.Timestamp]:
    """Get the maximum date for a given ticker in the existing Parquet dataset. If it doesn't exist, return None."""
    if not out_dir.exists():
        return None
    dset = ds.dataset(out_dir, format="parquet", partitioning="hive")
    tbl = dset.to_table(filter=ds.field("ticker") == ticker, columns=["date"])
    if tbl.num_rows == 0:
        return None
    pdf = tbl.to_pandas()
    return pd.to_datetime(pdf["date"]).max()


def fetch_ticker_data(ticker: str,
                      ts: TimeSeries,
                      output_size: str,
                      max_retries: int = 1, # to save api call quota
                      base_sleep: float = 2.0
) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV data for a given ticker from Alpha Vantage API with retries and jitter and returns DataFrame or None."""
    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"[{ticker}] Requesting {output_size} data, attempt {attempt}/{max_retries}...")
            df, meta_data = ts.get_daily(symbol=ticker, outputsize=output_size)
            df = df.rename(columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume",
            })
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            df = df.sort_index()

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["open", "high", "low", "close"])
            df["ticker"] = ticker

            logging.info(f"[{ticker}] Received {len(df):,} rows ({df.index.min().date()} → {df.index.max().date()}).")
            return df
        
        except Exception as e:
            wait = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.5)  # jitter
            logging.warning(f"[{ticker}] Error: {e} — backing off {wait:.1f}s")
            time.sleep(wait)
    
    logging.error(f"[{ticker}] Failed to fetch data after {max_retries} attempts.")
    return None


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    expected = ["open", "high", "low", "close", "volume", "ticker"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Ensure unique (ticker, date) pairs, sorted by date
    df = df[expected].copy()
    df["ticker"] = df["ticker"].astype("category")
    df = df[~df.index.duplicated(keep="last")]
    df = df.reset_index().drop_duplicates(subset=["ticker", "date"]).set_index("date").sort_index()
    return df


def to_arrow_table(df: pd.DataFrame) -> pa.Table:
    df.index = df.index.astype("datetime64[ms]")
    schema = pa.schema([
        pa.field("date", pa.timestamp("ms")),
        pa.field("ticker", pa.dictionary(index_type=pa.int32(), value_type=pa.string())),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.float64()),
    ])
    return pa.Table.from_pandas(df.reset_index(), schema=schema, preserve_index=False)


def write_partitioned_parquet(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = to_arrow_table(df)
    pq.write_to_dataset(
        table,
        root_path=out_dir,
        partition_cols=["ticker"],
        compression="zstd",
        existing_data_behavior="overwrite_or_ignore"
    )


def write_metadata(out_dir: Path, tickers: List[str], mode: str) -> None:
    meta = {
        "tickers": tickers,
        "mode": mode,
        "source": "Alpha Vantage",
        "run_at": datetime.now(UTC).isoformat() + "Z",
        "script": "download_data.py",
    }
    (Path(out_dir) / "_metadata.json").write_text(json.dumps(meta, indent=2))


# ---------------- Main ----------------

def main():
    # parse CLI args and setup logging
    args = parse_args()
    setup_logging(verbose=args.verbose)

    # get Alpha Vantage API key from .env, AV class for API calls
    api_key = get_api_key()
    ts = TimeSeries(key=api_key, output_format="pandas")

    # load tickers
    tickers = load_universe(Path(args.universe_file)) if args.tickers is None else args.tickers
    logging.info(f"Universe contains {len(tickers)} tickers.")

    out_dir = Path(args.out_dir)
    today = pd.Timestamp.today().normalize()

    total_new_rows = 0

    for i, ticker in enumerate(tickers, 1):
        last_date = None if args.mode == "init" else get_existing_max_date_for_ticker(out_dir, ticker)
        delta_bdays = len(pd.bdate_range(last_date + pd.Timedelta(days=1), today)) if last_date else None

        # figuring out which data acquisition mode to use
        size = "full"
        if args.mode == "update" and last_date is None:
            logging.info(f"No existing data for {ticker}, switching to init mode.")
        elif args.mode == "update" and delta_bdays > 90:
            logging.info(f"Last recorded entry for {ticker} is too old for update ({delta_bdays} days), switching to init mode.")
        elif args.mode == "update":
            size = "compact"

        # attempting to fetch given ticker data
        df = fetch_ticker_data(ticker, ts, size)
        if df is None or df.empty:
            logging.warning(f"[{ticker}] No data returned; skipping.")
            if i < len(tickers):
                time.sleep(15) # 5 requests per minute limit for Alpha Vantage
            continue

        # enforcing correct df schema
        df = enforce_schema(df)

        # in update mode, append only rows strictly newer than last_date
        if args.mode == "update" and last_date is not None:
            before = len(df)
            df = df[df.index > last_date]
            logging.info(f"[{ticker}] Filtering rows newer than {last_date.date()} → {len(df)} of {before}.")
            if df.empty:
                logging.info(f"[{ticker}] Up-to-date.")
                if i < len(tickers):
                    time.sleep(15) # 5 requests per minute limit for Alpha Vantage
                continue

        # write new data to Parquet dataset
        write_partitioned_parquet(df, out_dir)
        total_new_rows += len(df)

        if i < len(tickers):
            time.sleep(15) # 5 requests per minute limit for Alpha Vantage

    write_metadata(out_dir, tickers, args.mode)
    logging.info(f"Done. Wrote {total_new_rows:,} new rows to {out_dir}.")





if __name__ == "__main__":
    main()