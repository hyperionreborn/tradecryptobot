import argparse
import json
import time

from stocks.data_fetch import dataset_dir_name, make_dataset
from stocks.predict_now import predict_now
from stocks.train_model import TrainAll


if __name__ == "__main__":
    default_symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]

    parser = argparse.ArgumentParser(description="Daily stock prediction pipeline")
    parser.add_argument("--data_fetch", action="store_true", help="Fetch/build stock dataset")
    parser.add_argument("--train", action="store_true", help="Train stock model")
    parser.add_argument("--predict_now", action="store_true", help="Run one-shot daily prediction")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Ticker (e.g. AAPL) or 'all'")
    parser.add_argument("--years", type=int, default=12, help="History depth in years")
    parser.add_argument("--window_days", type=int, default=60, help="Lookback window in trading days")
    parser.add_argument("--horizon_days", type=int, default=5, help="Forecast horizon in trading days")
    parser.add_argument("--step", type=int, default=1, help="Stride between windows")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--cutoff_days", type=int, default=0, help="Shift end-date backwards by N days")
    parser.add_argument(
        "--no_regime_features",
        action="store_true",
        help="Disable regime feature engineering when building dataset",
    )
    parser.add_argument(
        "--label_mode",
        type=str,
        default="vol_scaled",
        choices=["price_direction", "vol_scaled"],
        help="Classification label mode for training",
    )
    parser.add_argument(
        "--label_atr_mult",
        type=float,
        default=0.35,
        help="ATR%% multiplier for vol_scaled label mode",
    )
    parser.add_argument(
        "--label_floor_pct",
        type=float,
        default=0.40,
        help="Minimum threshold in percent for vol_scaled label mode",
    )
    parser.add_argument(
        "--label_cap_pct",
        type=float,
        default=1.20,
        help="Maximum threshold in percent for vol_scaled label mode",
    )
    args = parser.parse_args()

    symbols = default_symbols if args.symbol == "all" else [args.symbol]

    if args.data_fetch:
        for s in symbols:
            print(f"Building dataset for {s}...")
            make_dataset(
                symbol=s,
                years=args.years,
                window_days=args.window_days,
                horizon_days=args.horizon_days,
                step=args.step,
                cutoff_days=args.cutoff_days,
                use_regime_features=(not args.no_regime_features),
            )
            time.sleep(0.5)

    if args.train:
        summary = {}
        for s in symbols:
            ds_dir = dataset_dir_name(s, args.window_days, args.horizon_days)
            acc, rmse = TrainAll(
                dataset_dir=ds_dir,
                EPOCHS=args.epochs,
                label_mode=args.label_mode,
                label_atr_mult=args.label_atr_mult,
                label_floor_pct=args.label_floor_pct,
                label_cap_pct=args.label_cap_pct,
            )
            summary[s] = {"accuracy": acc, "rmse_change": rmse}

        if args.symbol == "all":
            with open("stock_model_info.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print("Saved stock_model_info.json")

    if args.predict_now:
        if args.symbol == "all":
            for s in symbols:
                predict_now(s, args.window_days, args.horizon_days)
        else:
            predict_now(args.symbol, args.window_days, args.horizon_days)
