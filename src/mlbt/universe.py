# src/mlbt/universe.py
from pathlib import Path
import pandas as pd


def load_universe(path: Path) -> list[str]:
    """Load a universe CSV file (with a 'symbol' column) or text file (one ticker per line) and return a list of tickers."""
    if not path.exists():
        raise FileNotFoundError(f"Tickers file not found: {path}")
    
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if 'symbol' not in df.columns:
            raise ValueError(f"CSV file {path} must contain a 'symbol' column.")
        tickers = df['symbol'].dropna().astype(str).str.strip().unique().tolist()
    else:
        tickers = [line.strip() for line in path.read_text().splitlines() if line.strip()]         
    return tickers