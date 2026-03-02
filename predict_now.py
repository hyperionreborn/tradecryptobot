"""
Immediate one-shot live prediction — no waiting for candle boundaries.
Usage:
    python predict_now.py [--symbol BTCUSDT] [--window_days 14] [--resample_hours 4] [--horizon 3]
"""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import torch
import joblib

from crypto.model import ImprovedLSTMModel
from crypto.data_fetch import get_evaluate_window, scale_live_window


def predict_now(symbol: str, window_days: int, resample_hours: int, horizon: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = Path(f"{symbol}_{window_days}_{resample_hours}_{horizon}")
    model_path = dataset_dir / "42_binary_model.pt"
    scaler_path = dataset_dir / "scaler.pkl"
    config_path = dataset_dir / "model_config.json"

    if not model_path.exists():
        print(f"[ERROR] No trained model found at {model_path}")
        print("  Run: python main.py --train --symbol BTCUSDT --window_days 14 --resample_hours 4 --horizon 3")
        return

    with open(config_path) as f:
        config = json.load(f)

    model = ImprovedLSTMModel(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scaler = joblib.load(scaler_path)

    print(f"Fetching latest {window_days}-day window for {symbol}...")
    X_raw = get_evaluate_window(symbol, window_days, resample_hours)
    X = scale_live_window(X_raw, scaler).to(device)

    with torch.no_grad():
        logits, pred_change = model(X)
        prob_up = torch.sigmoid(logits).item()
        change_pct = pred_change.item() * 100

    horizon_hours = horizon * resample_hours
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print()
    print("=" * 48)
    print(f"  LIVE PREDICTION  |  {now_utc}")
    print("=" * 48)
    print(f"  Symbol          : {symbol}")
    print(f"  Forecast horizon: {horizon_hours}h ahead")
    print(f"  Prob UP         : {prob_up:.1%}")
    print(f"  Prob DOWN       : {1 - prob_up:.1%}")
    print(f"  Predicted change: {change_pct:+.2f}%")
    print()

    if prob_up >= 0.60:
        signal = "LONG  (strong)"
    elif prob_up >= 0.52:
        signal = "LONG  (weak)"
    elif prob_up <= 0.40:
        signal = "SHORT (strong)"
    elif prob_up <= 0.48:
        signal = "SHORT (weak)"
    else:
        signal = "NEUTRAL — no clear edge"

    print(f"  Signal          : {signal}")
    print("=" * 48)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-shot live prediction")
    parser.add_argument("--symbol",        type=str, default="BTCUSDT")
    parser.add_argument("--window_days",   type=int, default=14)
    parser.add_argument("--resample_hours",type=int, default=4)
    parser.add_argument("--horizon",       type=int, default=3)
    args = parser.parse_args()

    predict_now(args.symbol, args.window_days, args.resample_hours, args.horizon)
