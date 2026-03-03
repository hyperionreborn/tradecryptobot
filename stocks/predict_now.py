import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import torch

from .data_fetch import dataset_dir_name, get_evaluate_window, scale_live_window
from .model import ImprovedLSTMModel


def _magnitude_label(change_pct_abs: float) -> str:
    if change_pct_abs < 0.5:
        return "low"
    if change_pct_abs < 1.5:
        return "medium"
    return "high"


def predict_now(symbol: str, window_days: int, horizon_days: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = Path(dataset_dir_name(symbol, window_days, horizon_days))
    model_path = dataset_dir / "42_stock_model.pt"
    scaler_path = dataset_dir / "scaler.pkl"
    config_path = dataset_dir / "model_config.json"

    if not model_path.exists():
        print(f"[ERROR] Missing model at {model_path}")
        print(
            f"Run first: python stocks_main.py --train --symbol {symbol} "
            f"--window_days {window_days} --horizon_days {horizon_days}"
        )
        return

    with open(config_path, "r", encoding="utf-8") as f:
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
    X_raw = get_evaluate_window(symbol=symbol, window_days=window_days)
    X = scale_live_window(X_raw, scaler).to(device)

    with torch.no_grad():
        logits, pred_change = model(X)
        prob_up = torch.sigmoid(logits).item()
        change_pct = pred_change.item() * 100.0

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    magnitude = _magnitude_label(abs(change_pct))
    direction = "UP" if change_pct >= 0 else "DOWN"

    if prob_up >= 0.60:
        signal = "LONG (strong confidence)"
    elif prob_up >= 0.52:
        signal = "LONG (weak confidence)"
    elif prob_up <= 0.40:
        signal = "SHORT (strong confidence)"
    elif prob_up <= 0.48:
        signal = "SHORT (weak confidence)"
    else:
        signal = "NEUTRAL"

    print()
    print("=" * 60)
    print(f" STOCK DAILY PREDICTION | {now_utc}")
    print("=" * 60)
    print(f" Symbol             : {symbol}")
    print(f" Horizon            : {horizon_days} trading day(s)")
    print(f" Prob UP            : {prob_up:.1%}")
    print(f" Prob DOWN          : {1 - prob_up:.1%}")
    print(f" Predicted move     : {change_pct:+.2f}% ({direction})")
    print(f" Move strength      : {magnitude}")
    print(f" Signal             : {signal}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-shot daily stock prediction")
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--window_days", type=int, default=60)
    parser.add_argument("--horizon_days", type=int, default=5)
    args = parser.parse_args()
    predict_now(args.symbol, args.window_days, args.horizon_days)
