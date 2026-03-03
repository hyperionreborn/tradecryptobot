import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import torch

from .data_fetch import (
    dataset_dir_name,
    get_evaluate_window,
    get_evaluate_window_at_date,
    scale_live_window,
)
from .model import ImprovedLSTMModel


def _magnitude_label(change_pct_abs: float) -> str:
    if change_pct_abs < 0.5:
        return "low"
    if change_pct_abs < 1.5:
        return "medium"
    return "high"


def _load_dataset_config(dataset_dir: Path) -> dict:
    cfg_path = dataset_dir / "dataset_config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"use_regime_features": True}


def predict_now(
    symbol: str,
    window_days: int,
    horizon_days: int,
    predict_at_date: str | None = None,
    compare_realized: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = Path(dataset_dir_name(symbol, window_days, horizon_days))
    model_path = dataset_dir / "42_stock_model.pt"
    scaler_path = dataset_dir / "scaler.pkl"
    config_path = dataset_dir / "model_config.json"
    label_config_path = dataset_dir / "label_config.json"

    if not model_path.exists():
        print(f"[ERROR] Missing model at {model_path}")
        print(
            f"Run first: python stocks_main.py --train --symbol {symbol} "
            f"--window_days {window_days} --horizon_days {horizon_days}"
        )
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if label_config_path.exists():
        with open(label_config_path, "r", encoding="utf-8") as f:
            label_config = json.load(f)
    else:
        label_config = {
            "label_mode": "price_direction",
            "label_atr_mult": 0.35,
            "label_floor_pct": 0.40,
            "label_cap_pct": 1.20,
        }
    dataset_config = _load_dataset_config(dataset_dir)
    include_regime_features = bool(dataset_config.get("use_regime_features", True))

    model = ImprovedLSTMModel(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scaler = joblib.load(scaler_path)
    replay_target_date = None
    replay_as_of_date = None
    as_of_close = None
    realized_close = None
    if predict_at_date:
        X_raw, replay_as_of_date, replay_target_date, as_of_close, realized_close = get_evaluate_window_at_date(
            symbol=symbol,
            window_days=window_days,
            as_of_date=predict_at_date,
            horizon_days=horizon_days,
            include_regime_features=include_regime_features,
        )
    else:
        X_raw = get_evaluate_window(
            symbol=symbol,
            window_days=window_days,
            include_regime_features=include_regime_features,
        )
    X = scale_live_window(X_raw, scaler).to(device)

    with torch.no_grad():
        logits, pred_change = model(X)
        prob_up = torch.sigmoid(logits).item()
        change_pct = pred_change.item() * 100.0

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    magnitude = _magnitude_label(abs(change_pct))
    direction = "UP" if change_pct >= 0 else "DOWN"

    dynamic_up_threshold_pct = None
    if label_config.get("label_mode") == "vol_scaled":
        try:
            features_path = dataset_dir / "features.json"
            with open(features_path, "r", encoding="utf-8") as f:
                feature_names = json.load(f)
            atr_idx = feature_names.index("ATR_14")
            close_idx = feature_names.index("Close")
            atr_now = float(abs(X_raw[-1, atr_idx]))
            close_now = float(X_raw[-1, close_idx]) if float(X_raw[-1, close_idx]) != 0.0 else 1.0
            atr_pct = atr_now / close_now
            raw_t = float(label_config.get("label_atr_mult", 0.35)) * atr_pct * 100.0
            floor_t = float(label_config.get("label_floor_pct", 0.40))
            cap_t = float(label_config.get("label_cap_pct", 1.20))
            dynamic_up_threshold_pct = max(floor_t, min(cap_t, raw_t))
        except Exception:
            dynamic_up_threshold_pct = None

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
    if replay_as_of_date is not None:
        print(f" Replay as-of date  : {replay_as_of_date.date()}")
    if replay_target_date is not None:
        print(f" Target date        : {replay_target_date.date()}")
    print(f" Prob UP            : {prob_up:.1%}")
    print(f" Prob DOWN          : {1 - prob_up:.1%}")
    if dynamic_up_threshold_pct is not None:
        print(f" UP threshold today : +{dynamic_up_threshold_pct:.2f}% (vol-scaled)")
    print(f" Predicted move     : {change_pct:+.2f}% ({direction})")
    if compare_realized and (as_of_close is not None) and (realized_close is not None):
        realized_change_pct = ((realized_close / as_of_close) - 1.0) * 100.0
        realized_dir = "UP" if realized_change_pct >= 0 else "DOWN"
        print(f" Realized move      : {realized_change_pct:+.2f}% ({realized_dir})")
    elif compare_realized and predict_at_date:
        print(" Realized move      : N/A (future trading data unavailable for selected date)")
    print(f" Move strength      : {magnitude}")
    print(f" Signal             : {signal}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-shot daily stock prediction")
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--window_days", type=int, default=60)
    parser.add_argument("--horizon_days", type=int, default=5)
    parser.add_argument("--predict_at_date", type=str, default=None, help="Replay date YYYY-MM-DD")
    parser.add_argument("--compare_realized", action="store_true", help="Show realized move for replay mode")
    args = parser.parse_args()
    predict_now(
        args.symbol,
        args.window_days,
        args.horizon_days,
        predict_at_date=args.predict_at_date,
        compare_realized=args.compare_realized,
    )
