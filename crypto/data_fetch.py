import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf

# Try to import config for data directories (optional)
try:
    from .config import LSTM_DATA_DIR, WALLET_DATA_DIR
except ImportError:
    LSTM_DATA_DIR = None
    WALLET_DATA_DIR = None


def download_data(symbol: str, months: int, interval: str = "1h"):
    from datetime import date, timedelta
    import pandas as pd
    import numpy as np
    import yfinance as yf

    print(f"Downloading data for {symbol}...")

    # Be explicit to avoid surprises from future defaults
    if months == -1:
        df = yf.download(
            symbol,
            period="max",
            interval=interval,
            auto_adjust=False,
            group_by="column",
            progress=False,
            prepost=False,
        )
    else:
        end_date = date.today()
        start_date = end_date - timedelta(days=int(round(months * 30.44)))
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=False,
            group_by="column",
            progress=False,
            prepost=False,
        )

    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {symbol}")

    # If MultiIndex columns (e.g., (symbol, field) or (field,)) try to flatten
    original_columns = df.columns
    if isinstance(df.columns, pd.MultiIndex):
        # Build flattened names from tuples but try to pick the OHLCV part if present
        flattened = []
        for tup in df.columns:
            # Convert tuple elements to strings, skip empty/None
            parts = [str(x) for x in tup if x is not None and str(x) != ""]
            # Prefer any part that looks like an OHLCV field
            chosen = None
            for p in parts:
                pl = p.lower()
                if pl in ("open", "high", "low", "close", "volume", "adj close", "adjclose"):
                    chosen = p
                    break
            if chosen is None:
                # If none matches, try to drop the symbol (common in (symbol, field))
                if len(parts) >= 2:
                    # prefer the last part (often the field)
                    chosen = parts[-1]
                elif len(parts) == 1:
                    chosen = parts[0]
                else:
                    chosen = ""
            flattened.append(chosen)
        df = df.copy()
        df.columns = flattened

    # Normalize columns to Title case OHLCV. Allow variations like 'Adj Close' and 'adjclose'
    col_names = [str(c) for c in df.columns]
    col_map = {c.lower(): c for c in col_names}
    wanted = ["open", "high", "low", "close", "volume"]

    # Direct presence check
    missing = [w for w in wanted if w not in col_map]

    # Try best-effort substring matching for common variants (e.g., 'adj close', 'adjclose')
    if missing:
        for w in missing.copy():
            found = None
            for c in col_names:
                cl = c.lower().replace(" ", "")
                if w.replace(" ", "") in cl:
                    found = c
                    break
            if found:
                col_map[w] = found
                missing.remove(w)

    # Final fallback: try to detect columns by inspecting original MultiIndex tuples (if any)
    if missing and isinstance(original_columns, pd.MultiIndex):
        # Look through original tuples to find elements that match wanted names
        for w in missing.copy():
            candidate = None
            for tup in original_columns:
                for part in tup:
                    try:
                        pl = str(part).lower()
                    except Exception:
                        continue
                    if w in pl:
                        candidate = str(part)
                        break
                if candidate:
                    break
            if candidate:
                col_map[w] = candidate
                missing.remove(w)

    if missing:
        # If still missing, raise with a helpful hint showing what we actually got
        raise KeyError(
            f"Expected OHLCV columns missing: {missing}. Got columns={list(df.columns)}. "
            f"Tip: ensure auto_adjust=False and group_by='column' (yfinance output can vary)."
        )

    df = df[[col_map[w] for w in wanted]].copy()
    df.columns = ["Open", "High", "Low", "Close", "Volume"]

    # Ensure DatetimeIndex (sometimes it’s plain Index)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    # Strip timezone if present (resample expects naive or consistent tz)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    df = df.sort_index().dropna(how="any")
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
    # Resample to specified interval
    df_resampled = df.resample(f'{resample_hours}h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    # Local ATH/ATL features
    df_resampled['local_ATH'] = df_resampled['Close'].cummax()
    df_resampled['local_ATL'] = df_resampled['Close'].cummin()
    
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


def make_dataset(
    symbol: str,
    months: int,
    window_days: int,
    resample_hours: int,
    horizon: int,
    step: int,
    outdir: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Orchestrate dataset creation: download, compute features, build windows, and save.
    
    Returns:
        Tuple of (X, y, feature_names)
    """
    print(f"Downloading data for {symbol}...")
    df_raw = download_data(symbol, months)
    print(f"Downloaded {len(df_raw)} hourly samples")
    
    print(f"Computing features with {resample_hours}h resampling...")
    df_features, feature_names = compute_features(df_raw, resample_hours)
    print(f"Resampled to {len(df_features)} samples with {len(feature_names)} features")
    
    print(f"Building sliding windows (window={window_days} days, horizon={horizon}, step={step})...")
    X, y = build_windows(df_features, window_days, resample_hours, horizon, step)
    print(f"Created {len(X)} windows with shape {X.shape}")
    
    # Create output directory
    outdir_path = Path(outdir)
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
    parser.add_argument(
        "--outdir",
        type=str,
        default="./dataset",
        help="Output directory for X.npy, y.npy, features.json"
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
            outdir=args.outdir
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


def convert(
    collected_data,
    features=[
        "sells1m",
        "buys1m",
        "swaps1m",
        "age",
        "volume",
        "price1m",
        "buy_sell_ratio",
        "price5m",
        "liquidity",
        "holders",
        "top10holders",
    ],
):
    """Convert collected token data to LSTM format (kept for backward compatibility)."""
    lstm_data = {}

    for token_address, raw_data in collected_data.items():
        try:
            # Validate data before conversion
            if not raw_data or not isinstance(raw_data, list):
                print(f"Invalid data format for token {token_address} \n")
                continue

            X = []
            for item in raw_data:
                try:
                    row = []
                    for feature in features:
                        value = item.get(feature, 0.0)
                        # Ensure we have a numeric value
                        row.append(float(value) if value is not None else 0.0)
                    X.append(row)
                except (ValueError, TypeError) as e:
                    print(f"Error processing row for token {token_address}: {str(e)} \n")
                    continue

            if len(X) > 0:
                X = np.array(X, dtype=np.float32)
                if X.shape[1] == len(features):
                    lstm_data[token_address] = X
                else:
                    print(f"Incorrect feature count for token {token_address} \n")

        except Exception as e:
            print(f"Error converting data for token {token_address}: {str(e)} \n")
            continue

    return lstm_data


def generate_labels(collected_data, threshold=0.0):
    """Generate labels from collected data (kept for backward compatibility)."""
    labels = {}
    for token_address, data_points in collected_data.items():
        try:
            price_now = float(data_points[-2]["price1m"])
            price_later = float(data_points[-1]["price1m"])
            change = (price_later - price_now) / price_now

            labels[token_address] = np.array(int(change > threshold))
        except (TypeError, KeyError, ZeroDivisionError):
            continue

    return labels


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

def data_get(change=1, hours_collect=1):
    """
    Collect token data and generate labels.
    
    Args:
        change: Number of hours to wait for price change
        hours_collect: Number of hours of data to collect
    
    Returns:
        Tuple of (lstm_data dict, labels dict)
    """
    raise NotImplementedError(
        "data_get() needs to be re-implemented. This function was part of the "
        "original API-based token data collection system that was removed during "
        "the stock data preparation refactoring."
    )


def json_get_merged_tokens(filename):
    """
    Load merged token data from JSON file.
    
    Args:
        filename: Name of the JSON file (without extension or full path)
    
    Returns:
        Dictionary of token data or None if file not found
    """
    try:
        filepath = Path(filename)
        if not filepath.suffix:
            filepath = filepath.with_suffix('.json')
        
        # If not absolute path, check multiple locations
        if not filepath.is_absolute():
            # First check in LSTM_DATA_DIR if configured
            if LSTM_DATA_DIR:
                alt_path = Path(LSTM_DATA_DIR) / filepath
                if alt_path.exists():
                    filepath = alt_path
            # If still not found, check current directory (filepath as-is)
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading token data from {filename}: {e}")
        return None


def take_snapshot():
    """
    Take a snapshot of current tokens from the API.
    
    Returns:
        List of token dictionaries with token information
    """
    raise NotImplementedError(
        "take_snapshot() needs to be re-implemented. This function should fetch "
        "current token data from your API (e.g., DexScreener)."
    )


def collect_data_for_tokens(tokens, max_data_points=60):
    """
    Collect historical data for a list of tokens.
    
    Args:
        tokens: List of token dictionaries
        max_data_points: Maximum number of data points to collect per token
    
    Returns:
        Dictionary mapping token addresses to lists of data points
    """
    raise NotImplementedError(
        "collect_data_for_tokens() needs to be re-implemented. This function should "
        "collect historical data for tokens from your API."
    )


def json_get_merged_wallets(filename):
    """
    Load merged wallet data from JSON file.
    
    Args:
        filename: Name of the JSON file (without extension or full path)
    
    Returns:
        Dictionary of wallet data or None if file not found
    """
    try:
        filepath = Path(filename)
        if not filepath.suffix:
            filepath = filepath.with_suffix('.json')
        
        # If not absolute path, check multiple locations
        if not filepath.is_absolute():
            # First check in WALLET_DATA_DIR if configured
            if WALLET_DATA_DIR:
                alt_path = Path(WALLET_DATA_DIR) / filepath
                if alt_path.exists():
                    filepath = alt_path
            # If still not found, check current directory (filepath as-is)
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading wallet data from {filename}: {e}")
        return None


def get_wallet_buys(*args, **kwargs):
    """
    Get buy transactions for wallets.
    
    Returns:
        Dictionary or list of wallet buy data
    """
    raise NotImplementedError(
        "get_wallet_buys() needs to be re-implemented. This function should fetch "
        "wallet buy transaction data from your API."
    )


def json_get(path):
    """
    Load JSON data from file path.
    
    Args:
        path: Path to JSON file (can be relative or absolute)
    
    Returns:
        Parsed JSON data or None if file not found
    """
    try:
        filepath = Path(path)
        
        # If not absolute path, check in LSTM_DATA_DIR if configured
        if not filepath.is_absolute():
            if LSTM_DATA_DIR:
                alt_path = Path(LSTM_DATA_DIR) / filepath
                if alt_path.exists():
                    filepath = alt_path
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading JSON from {path}: {e}")
        return None


if __name__ == "__main__":
    main()
