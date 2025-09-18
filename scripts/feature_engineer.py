import pandas as pd
from pathlib import Path


def calculate_features(df):
    """
    Calculates financial features from raw OHLCV data.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'ticker']
    
    Returns:
        pd.DataFrame: DataFrame with original columns plus new feature columns.
    """
    grouped = df.groupby('ticker')

    print(grouped)

    # Calculate simple past returns (Momentum)
    df['returns_5d'] = grouped['close'].transform(lambda x: x.pct_change(5))
    df['returns_21d'] = grouped['close'].transform(lambda x: x.pct_change(21))
    
    # Calculate rolling volatility
    df['volatility_21d'] = grouped['close'].transform(lambda x: x.pct_change().rolling(21).std())
    
    # Calculate volume ratio (current volume / average volume)
    df['volume_ratio'] = df['volume'] / grouped['volume'].transform(lambda x: x.rolling(20).mean())
    
    # Add more features here later depending on strategy (e.g., RSI, MACD)
    
    df = df.dropna()
    
    return df  


def main():
    data_path = Path(__file__).parent.parent / 'data' / 'equity_data.parquet'
    print("Loading raw data...")
    raw_df = pd.read_parquet(data_path)
    print(f"Loaded {len(raw_df)} rows of raw data.")
    
    print("Calculating features...")
    featured_df = calculate_features(raw_df)
    print(f"Feature engineering complete. {len(featured_df)} rows remaining after calculating features.")

    output_path = Path(__file__).parent.parent / 'data' / 'featured_data.parquet'
    featured_df.to_parquet(output_path)
    print(f"Featured data saved to {output_path}")
    
    # Quick inspection
    print("\nFeature Preview:")
    print(featured_df.tail(10))



if __name__ == "__main__":
    main()