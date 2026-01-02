import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from binance.client import Client
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler
import joblib


# Try to import config for data directories (optional)

client = Client()
def get_tradable_futures_symbols():
    info = client.futures_exchange_info()
    symbols = []

    for s in info["symbols"]:
        if (
            s["status"] == "TRADING"
            and s["contractType"] == "PERPETUAL"
            and s["quoteAsset"] == "USDT"
        ):
            try:
                klines = client.futures_klines(
                    symbol=s["symbol"],
                    interval=Client.KLINE_INTERVAL_1HOUR,
                    limit=1
                )
                if klines:
                    symbols.append(s["symbol"])
            except:
                pass

    return symbols
def download_data(symbol: str, months: int, interval: str = "1h"):


    print(f"Downloading data for {symbol}...")

    # Be explicit to avoid surprises from future defaults
    if months == -1:
        # Full history: Start from Binance's earliest BTCUSDT data
        start_str = "1 Aug 2017"  # ~1502928000000 ms
    else:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=int(round(months * 30.44)))  # ~30.44 days/month
        start_str = start_time.strftime("%d %b %Y")

        # Fetch with auto-pagination (handles limits)
    klines = client.get_historical_klines(
        symbol,
        interval,
        start_str=start_str
    )
    columns_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 'Number of Trades',
                          'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']

    df = pd.DataFrame(klines, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
                                         'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume',
                                         'Taker Buy Quote Asset Volume', 'Ignore'])

    for col in columns_to_convert:
        df[col] = df[col].astype(float)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {symbol}")



    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms', utc=True)
    df = df.set_index('Open Time').sort_index()
    # Ensure DatetimeIndex (sometimes it’s plain Index)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    # Strip timezone if present (resample expects naive or consistent tz)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    df = df.sort_index().dropna(how="any")
    df = df.drop(columns=['Ignore','Close Time'])
    print(f"Downloaded {len(df)} {interval} samples")
    return df


def compute_features(df: pd.DataFrame, resample_hours: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    Resample data and compute technical indicators once on the full dataset.
    
    Args:
        df: Raw OHLCV DataFrame with hourly data
        resample_hours: Resampling interval in hours
    
    Returns:
        Tuple of (feature DataFrame, feature column names)
    """
    df_hours = len(df)
    # Resample to specified interval
    df_resampled = df.resample(f'{resample_hours}h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'Quote Asset Volume':'sum',
        'Number of Trades':'sum',
        'Taker Buy Base Asset Volume':'sum',
        'Taker Buy Quote Asset Volume':'sum'


    }).dropna()
    # Local ATH/ATL features
    df_resampled['local_ATH'] = df_resampled['Close'].cummax()
    df_resampled['local_ATL'] = df_resampled['Close'].cummin()
    df_resampled['pct_change'] = df_resampled['Close'].pct_change(12)
    is_ath = df_resampled['Close'] == df_resampled['local_ATH']
    is_atl = df_resampled['Close'] == df_resampled['local_ATL']
    
    # Time since local high
    not_ath = ~is_ath
    cumsum_not_ath = not_ath.cumsum()
    last_ath_cumsum = cumsum_not_ath.where(is_ath).ffill().fillna(0)
    df_resampled['time_local_High'] = cumsum_not_ath - last_ath_cumsum
    
    # Time since local low
    not_atl = ~is_atl
    cumsum_not_atl = not_atl.cumsum()
    last_atl_cumsum = cumsum_not_atl.where(is_atl).ffill().fillna(0)
    df_resampled['time_local_Low'] = cumsum_not_atl - last_atl_cumsum
    
    # Temporal features
    df_resampled['day_of_week'] = df_resampled.index.dayofweek
    df_resampled['hour'] = df_resampled.index.hour
    
    # Technical indicators with safe NaN handling
    # EMA
    df_resampled['EMA_12'] = df_resampled['Close'].ewm(span=12, min_periods=1).mean()
    df_resampled['EMA_26'] = df_resampled['Close'].ewm(span=26, min_periods=1).mean()
    
    # MACD
    df_resampled['MACD'] = df_resampled['EMA_12'] - df_resampled['EMA_26']
    df_resampled['MACD_signal'] = df_resampled['MACD'].ewm(span=9, min_periods=1).mean()
    
    # RSI with safe division
    delta = df_resampled['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14, min_periods=1).mean()
    # Avoid divide by zero
    rs = gain / loss.replace(0, np.nan)
    rs = rs.fillna(0)
    df_resampled['RSI'] = 100 - (100 / (1 + rs))
    df_resampled['RSI'] = df_resampled['RSI'].fillna(50.0)  # Default to neutral RSI
    
    # Volatility
    df_resampled['Volatility'] = df_resampled['Close'].rolling(window=12, min_periods=1).std()
    df_resampled['Volatility'] = df_resampled['Volatility'].fillna(0.0)
    
    # Drop intermediate columns
    df_resampled = df_resampled.drop(columns=['local_ATH', 'local_ATL'])
    
    # Drop any remaining NaN rows
    df_resampled = df_resampled.dropna()
    
    # Get feature column names
    feature_columns = df_resampled.columns.tolist()
    
    return df_resampled, feature_columns


def build_windows(
    df_features: pd.DataFrame,
    window_days: int,
    resample_hours: int,
    horizon: int,
    step: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows and targets using vectorized operations.
    
    Args:
        df_features: Feature DataFrame (already resampled)
        window_days: Number of days in each window
        resample_hours: Resampling interval in hours
        horizon: Number of resampled steps ahead to predict
        step: Stride between sliding windows
    
    Returns:
        Tuple of (X: array of shape (N, T, F), y: array of shape (N,))
    """
    # Calculate window size in resampled steps
    steps_per_day = 24 / resample_hours
    window_size = int(window_days * steps_per_day)
    
    if window_size < 1:
        raise ValueError(f"Window size too small: {window_size} steps. Increase window_days or decrease resample_hours.")
    
    if len(df_features) < window_size + horizon:
        raise ValueError(f"Not enough data: {len(df_features)} samples, need at least {window_size + horizon}")
    
    # Convert to numpy array
    feature_array = df_features.values.astype(np.float32)
    num_features = feature_array.shape[1]
    
    # Extract Close prices for targets (assuming 'Close' is in the features)
    close_idx = df_features.columns.get_loc('Close')
    close_prices = feature_array[:, close_idx]
    
    # Calculate number of windows
    max_start = len(feature_array) - window_size - horizon + 1
    if max_start <= 0:
        raise ValueError(f"Not enough data: {len(feature_array)} samples, need at least {window_size + horizon}")
    
    # Generate all valid start indices with step
    start_indices = np.arange(0, max_start, step)
    
    if len(start_indices) == 0:
        raise ValueError(f"No valid windows can be created with the given parameters.")
    
    # Vectorized window creation using advanced indexing
    # Create index array for all windows: shape (num_windows, window_size)
    window_indices = start_indices[:, None] + np.arange(window_size)
    
    # Extract windows: shape (num_windows, window_size, num_features)
    X = feature_array[window_indices].astype(np.float32)
    
    # Targets: Close price horizon steps ahead
    target_indices = start_indices + window_size + horizon - 1
    # Ensure we don't go out of bounds
    valid_mask = target_indices < len(close_prices)
    X = X[valid_mask]
    target_indices = target_indices[valid_mask]
    
    y = close_prices[target_indices].astype(np.float32)
    
    return X, y

def get_evaluate_window(symbol:str,window_days:int,resample_hours:int):
    steps_per_day = 24 // resample_hours
    window_steps = window_days * steps_per_day
    df = download_data(symbol,(window_days*60))
    df_c = compute_features(df,resample_hours=resample_hours)
    window_df = df_c.iloc[-window_steps:]

    X = window_df.values.astype(np.float32)
    return X
    
def make_dataset(
    symbol: str,
    months: int,
    window_days: int,
    resample_hours: int,
    horizon: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Orchestrate dataset creation: download, compute features, build windows, and save.
    
    Returns:
        Tuple of (X, y, feature_names)
    """
    df_raw = download_data(symbol, months)
    print(f"Downloaded {len(df_raw)} hourly samples")
    
    print(f"Computing features with {resample_hours}h resampling...")
    df_features, feature_names = compute_features(df_raw, resample_hours)
    print(f"Resampled to {len(df_features)} samples with {len(feature_names)} features")
    
    print(f"Building sliding windows (window={window_days} days, horizon={horizon}, step={step})...")
    X, y = build_windows(df_features, window_days, resample_hours, horizon, step)
    print(f"Created {len(X)} windows with shape {X.shape}")
    
    # Create output directory
    outdir_path = Path(f"{symbol}_{window_days}_{resample_hours}_{horizon}")
    outdir_path.mkdir(parents=True, exist_ok=True)

    # Save arrays
    x_path = outdir_path / "X.npy"
    y_path = outdir_path / "y.npy"
    features_path = outdir_path / "features.json"
    
    print(f"Saving dataset to {outdir_path}...")
    np.save(x_path, X)
    np.save(y_path, y)
    
    # Save feature names as JSON
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    print(f"\nDataset saved successfully!")
    print(f"  X shape: {X.shape} -> {x_path}")
    print(f"  y shape: {y.shape} -> {y_path}")
    print(f"  Features: {len(feature_names)} -> {features_path}")
    print(f"  Feature names: {feature_names}")
    
    return X, y, feature_names


def main():
    """Parse CLI arguments and generate dataset."""
    parser = argparse.ArgumentParser(
        description="Generate LSTM-ready dataset from stock/crypto data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC-USD",
        help="Stock/crypto symbol (e.g., BTC-USD, AAPL)"
    )
    parser.add_argument(
        "--months",
        type=int,
        default=6,
        help="Number of past months to fetch (-1 for full history)"
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=14,
        help="Number of past days in each training window"
    )
    parser.add_argument(
        "--resample-hours",
        type=int,
        default=12,
        help="Resampling interval in hours"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Number of resampled steps ahead to predict"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Stride between sliding windows"
    )

    
    args = parser.parse_args()
    
    try:
        make_dataset(
            symbol=args.symbol,
            months=args.months,
            window_days=args.window_days,
            resample_hours=args.resample_hours,
            horizon=args.horizon,
            step=args.step,

        )
    except Exception as e:
        print(f"Error generating dataset: {e}")
        raise


# Keep these functions for backward compatibility with other modules
def safe_divide(numerator, denominator, default=1.0):
    """Helper function to safely divide two values."""
    try:
        num = float(numerator)
        den = float(denominator)
        if den <= 0:
            return default
        return num / den
    except (TypeError, ValueError, ZeroDivisionError):
        return default


def safe_float(value):
    """Helper function to safely convert values to float"""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        # For nested dictionaries, try to get the first numeric value
        for v in value.values():
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0





# Backward compatibility: keep get_training_data for existing code
def get_training_data(symbol, days, months=-1, interval=12):
    """
    Legacy function for backward compatibility.
    Use make_dataset() or main() for new code.
    """
    df = download_data(symbol=symbol, months=months)
    window_hours = days * 24
    resample_hours = interval
    
    df_features, feature_names = compute_features(df, resample_hours)
    
    # Use default horizon=1, step=1
    X, y = build_windows(df_features, days, resample_hours, horizon=1, step=1)
    
    return X, y, feature_names


# Token/Wallet data collection functions (stubs for backward compatibility)
# NOTE: These functions were removed during refactoring as they are part of a different
# API-based token data collection system. They need to be re-implemented based on
# your original API integration code.







if __name__ == "__main__":
    main()
