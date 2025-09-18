"""
Data acquisition module for fetching equity price data from Alpha Vantage.
Handles API communication, rate limiting, and local data saving.
"""

import os
import time
from pathlib import Path

import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv

# loading environment variables
load_dotenv()


def get_api_key():
    """Safely retrieves API key from environment variables."""
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment variables. Please check your .env file.")
    return api_key


def fetch_ticker_data(ticker, ts, output_size='compact'):
    """
    Fetches daily historical data for a single ticker with error handling.
    
    Parameters:
        ticker (str): The stock ticker symbol.
        ts (TimeSeries): Authenticated Alpha Vantage TimeSeries object.
        output_size (str): 'compact' (~100 days) or 'full' (~20 years).
    
    Returns:
        pd.DataFrame: DataFrame with OHLCV data for the ticker, or None if failed.
    """
    try:
        print(f"Fetching data for {ticker}...")
        data_df, meta_data = ts.get_daily(symbol=ticker, outputsize=output_size)
        data_df = data_df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        })
        data_df = data_df.sort_index()
        data_df = data_df.astype(float)
        data_df['ticker'] = ticker
        print(f"Successfully fetched {len(data_df)} days of data for {ticker}.")
        return data_df
    
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def main():
    """Main function to orchestrate the data download for a list of tickers."""
    api_key = get_api_key()
    ts = TimeSeries(key=api_key, output_format='pandas')

    tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META']
    # For a full backtest, you would eventually want a much larger list.
    # tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()

    all_data = []

    for ticker in tickers:
        df = fetch_ticker_data(ticker, ts, output_size='compact')
        if df is not None:
            all_data.append(df)
        # Alpha Vantage free tier has a rate limit of 5 API requests per minute.
        # This delay is CRUCIAL to avoid getting blocked.
        time.sleep(15) # 60 seconds / 5 requests = 12 sec/request. Using 15 for safety.
        
    if all_data:
        print("Combining data for all tickers...")
        full_df = pd.concat(all_data)

        # --- DATA TYPE CLEANING ---
        # 1. Ensure our numeric columns are actually floats
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            # pd.to_numeric converts to number, 'coerce' turns non-numbers into NaN
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
        
        # 2. The 'ticker' column should be a string (category is more efficient for repeated values)
        full_df['ticker'] = full_df['ticker'].astype('category')
        
        # 3. Drop any rows that might have become all NaNs due to the conversion
        full_df = full_df.dropna(subset=numeric_columns, how='all')
        # ------------------------------------------

        data_dir = Path(__file__).parent.parent / 'data'
        data_dir.mkdir(exist_ok=True)

        file_path = data_dir / 'equity_data.parquet'
        full_df.to_parquet(file_path)
        print(f"Data successfully saved to {file_path}")

        sample_file_path = data_dir / 'equity_data_sample.csv'
        full_df.tail(100).to_csv(sample_file_path)
        print(f"Sample data saved to {sample_file_path} for inspection.")

    else:
        print("No data was successfully downloaded.")



if __name__ == "__main__":
    main()
    