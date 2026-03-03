import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yfinance as yf


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    keep = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    return df[keep].copy()


def download_data(symbol: str, years: int, cutoff_days: int = 0) -> pd.DataFrame:
    end_date = datetime.utcnow().date() - timedelta(days=int(cutoff_days))
    start_date = end_date - timedelta(days=int(round(years * 365.25)))

    df = yf.download(
        tickers=symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {symbol}")

    df = _normalize_ohlcv_columns(df)
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df.dropna().sort_index()


def compute_features(df: pd.DataFrame, include_regime_features: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    feat = df.copy()
    eps = 1e-8

    feat["return_1d"] = feat["Close"].pct_change().fillna(0.0)
    feat["return_5d"] = feat["Close"].pct_change(5).fillna(0.0)
    feat["return_20d"] = feat["Close"].pct_change(20).fillna(0.0)
    feat["log_return"] = np.log(feat["Close"] / feat["Close"].shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    feat["EMA_12"] = feat["Close"].ewm(span=12, adjust=False, min_periods=1).mean()
    feat["EMA_26"] = feat["Close"].ewm(span=26, adjust=False, min_periods=1).mean()
    feat["MACD"] = feat["EMA_12"] - feat["EMA_26"]
    feat["MACD_signal"] = feat["MACD"].ewm(span=9, adjust=False, min_periods=1).mean()

    delta = feat["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + eps)
    feat["RSI"] = (100 - (100 / (1 + rs))).fillna(50.0)

    tr1 = feat["High"] - feat["Low"]
    tr2 = (feat["High"] - feat["Close"].shift(1)).abs()
    tr3 = (feat["Low"] - feat["Close"].shift(1)).abs()
    feat["ATR_14"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(window=14, min_periods=1).mean()

    feat["volatility_10d"] = feat["log_return"].rolling(window=10, min_periods=1).std().fillna(0.0)
    feat["volatility_20d"] = feat["log_return"].rolling(window=20, min_periods=1).std().fillna(0.0)
    feat["volatility_ratio"] = feat["volatility_10d"] / (feat["volatility_20d"] + eps)

    vol_mean = feat["Volume"].rolling(window=20, min_periods=1).mean()
    vol_std = feat["Volume"].rolling(window=20, min_periods=1).std().fillna(0.0)
    feat["volume_zscore"] = ((feat["Volume"] - vol_mean) / (vol_std + eps)).clip(-5, 5)

    feat["day_of_week_sin"] = np.sin(2 * np.pi * feat.index.dayofweek / 7)
    feat["day_of_week_cos"] = np.cos(2 * np.pi * feat.index.dayofweek / 7)

    bb_mid = feat["Close"].rolling(20, min_periods=1).mean()
    bb_std = feat["Close"].rolling(20, min_periods=1).std().fillna(0.0)
    feat["bb_upper"] = bb_mid + 2 * bb_std
    feat["bb_lower"] = bb_mid - 2 * bb_std
    feat["bb_width"] = (feat["bb_upper"] - feat["bb_lower"]) / (feat["Close"] + eps)

    if include_regime_features:
        trend_spread = (feat["EMA_12"] - feat["EMA_26"]) / (feat["Close"] + eps)
        feat["trend_strength"] = trend_spread.abs()
        feat["realized_vol_20d"] = feat["volatility_20d"] * np.sqrt(252.0)
        feat["trend_state"] = np.where(
            trend_spread > 0.0015,
            1.0,
            np.where(trend_spread < -0.0015, -1.0, 0.0),
        )

    feat = feat.drop(columns=["log_return", "EMA_26", "MACD_signal", "bb_upper", "bb_lower"])
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()

    feature_columns = feat.columns.tolist()
    return feat, feature_columns


def build_windows(
    df_features: pd.DataFrame,
    window_days: int,
    horizon_days: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    window_size = int(window_days)
    if window_size < 2:
        raise ValueError("window_days must be at least 2")
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")
    if len(df_features) < window_size + horizon_days:
        raise ValueError(
            f"Not enough data: {len(df_features)} rows, need at least {window_size + horizon_days}"
        )

    feature_array = df_features.values.astype(np.float32)
    close_idx = df_features.columns.get_loc("Close")
    close_prices = feature_array[:, close_idx]

    max_start = len(feature_array) - window_size - horizon_days + 1
    start_indices = np.arange(0, max_start, int(step))
    if len(start_indices) == 0:
        raise ValueError("No windows generated. Try a smaller window_days or horizon_days.")

    window_indices = start_indices[:, None] + np.arange(window_size)
    X = feature_array[window_indices].astype(np.float32)

    target_indices = start_indices + window_size + horizon_days - 1
    y = close_prices[target_indices].astype(np.float32)
    return X, y


def dataset_dir_name(symbol: str, window_days: int, horizon_days: int) -> str:
    safe = symbol.replace("/", "_").replace(".", "_")
    return f"{safe}_daily_{window_days}_{horizon_days}"


def make_dataset(
    symbol: str,
    years: int,
    window_days: int,
    horizon_days: int,
    step: int,
    cutoff_days: int = 0,
    use_regime_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df_raw = download_data(symbol=symbol, years=years, cutoff_days=cutoff_days)
    df_features, feature_names = compute_features(df_raw, include_regime_features=use_regime_features)
    X, y = build_windows(df_features, window_days, horizon_days, step)

    outdir_path = Path(dataset_dir_name(symbol, window_days, horizon_days))
    outdir_path.mkdir(parents=True, exist_ok=True)

    np.save(outdir_path / "X.npy", X)
    np.save(outdir_path / "y.npy", y)
    with open(outdir_path / "features.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    with open(outdir_path / "dataset_config.json", "w", encoding="utf-8") as f:
        json.dump({"use_regime_features": bool(use_regime_features)}, f, indent=2)

    return X, y, feature_names


def get_evaluate_window(
    symbol: str,
    window_days: int,
    years: int = 10,
    include_regime_features: bool = True,
) -> np.ndarray:
    df = download_data(symbol=symbol, years=max(years, 2), cutoff_days=0)
    feat, _ = compute_features(df, include_regime_features=include_regime_features)
    window_df = feat.iloc[-window_days:]
    if len(window_df) < window_days:
        raise ValueError(f"Not enough rows for inference window of {window_days} days")
    return window_df.values.astype(np.float32)


def _download_data_between(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    df = yf.download(
        tickers=symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {symbol} between {start_date.date()} and {end_date.date()}")
    df = _normalize_ohlcv_columns(df)
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df.dropna().sort_index()


def get_evaluate_window_at_date(
    symbol: str,
    window_days: int,
    as_of_date: str,
    horizon_days: int = 1,
    years: int = 10,
    include_regime_features: bool = True,
) -> Tuple[np.ndarray, pd.Timestamp, Optional[pd.Timestamp], float, Optional[float]]:
    anchor = pd.Timestamp(as_of_date).normalize()
    if pd.isna(anchor):
        raise ValueError(f"Invalid as_of_date: {as_of_date}")

    start_date = anchor - timedelta(days=int(round(max(years, 2) * 365.25)))
    # Add calendar-day buffer so we can access the future trading day(s) for realized comparison.
    end_date = anchor + timedelta(days=max(10, int(horizon_days) * 4 + 5))
    df = _download_data_between(symbol=symbol, start_date=start_date.to_pydatetime(), end_date=end_date.to_pydatetime())
    feat, _ = compute_features(df, include_regime_features=include_regime_features)
    if feat.empty:
        raise ValueError("No feature rows available for replay")

    past_mask = feat.index <= anchor
    if not past_mask.any():
        raise ValueError(f"No market data on or before {anchor.date()} for {symbol}")

    as_of_idx = int(np.where(past_mask)[0][-1])
    if as_of_idx + 1 < window_days:
        raise ValueError(
            f"Not enough history before {feat.index[as_of_idx].date()} for window_days={window_days}"
        )

    start_idx = as_of_idx - window_days + 1
    window_df = feat.iloc[start_idx : as_of_idx + 1]
    if len(window_df) != window_days:
        raise ValueError(f"Window length mismatch: expected {window_days}, got {len(window_df)}")

    as_of_actual = pd.Timestamp(feat.index[as_of_idx]).normalize()
    as_of_close = float(feat.iloc[as_of_idx]["Close"])

    target_idx = as_of_idx + int(horizon_days)
    target_date = None
    target_close = None
    if 0 <= target_idx < len(feat):
        target_date = pd.Timestamp(feat.index[target_idx]).normalize()
        target_close = float(feat.iloc[target_idx]["Close"])

    return window_df.values.astype(np.float32), as_of_actual, target_date, as_of_close, target_close


def scale_live_window(X: np.ndarray, scaler) -> torch.Tensor:
    t_steps, n_features = X.shape
    X_scaled = scaler.transform(X.reshape(-1, n_features)).reshape(1, t_steps, n_features)
    return torch.tensor(X_scaled, dtype=torch.float32)
