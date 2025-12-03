import time

import crypto
import argparse
import asyncio



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto prediction tool")
    parser.add_argument("--data_fetch", action="store_true", help="fetching data")
    parser.add_argument("--train", action="store_true", help="Training mode")
    parser.add_argument("--predict", action="store_true", help="Prediction mode")
    parser.add_argument("--test", type=str, help="Test predictions on saved data file")
    parser.add_argument("--window_days", type=int, default=14, help="Number of past days in each training window")
    parser.add_argument("--horizon", type=int, default=1, help="Number of resampled steps ahead to predict")
    parser.add_argument("--resample_hours", type=int, default=12, help="Resampling interval in hours")
    parser.add_argument("--months", type=int, default=6, help="Number of past months to fetch (-1 for full history)")
    parser.add_argument("--step", type=int, default=1, help="Stride between sliding windows")
    parser.add_argument("--outdir", type=str, default="./dataset", help="Output directory for dataset")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--backtest",action="store_true",help="backtesting")
    parser.add_argument("--test_nlp",action="store_true",help="test the nlp model")
    parser.add_argument("--symbol",type=str,default="BTCUSDT",help="pair to train on")

    args = parser.parse_args()
    crypto.load_config()
    ##.connect()
    # crypto.api_up()
    if args.data_fetch:
        crypto.make_dataset(
            symbol=args.symbol, # Defaulting to BTC-USD as per original intent or make it an arg? Adding symbol arg would be good too but sticking to requested ones first.
            months=args.months,
            window_days=args.window_days,
            resample_hours=args.resample_hours,
            horizon=args.horizon,
            step=args.step,
        )


    if args.train:
        # Prefer new dataset training if available
        crypto.TrainAll(dataset_dir=f"{args.symbol}_{args.window_days}_{args.resample_hours}_{args.horizon}", EPOCHS=args.epochs)
        # Fallback or alternative: crypto.TrainAll(hours_collect=int(args.hours_collect))

    if args.test_nlp:
        crypto.test_nlp()

