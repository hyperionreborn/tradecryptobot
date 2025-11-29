import time

import crypto
import argparse
import asyncio



if __name__ == "__main__":
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
        "--window_days",
        type=int,
        default=14,
        help="Number of past days in each training window"
    )
    parser.add_argument(
        "--resample_hours",
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
    parser.add_argument("--train", action="store_true", help="train the model")
    parser.add_argument("--data_fetch", action="store_true", help="train the model")
    args = parser.parse_args()
    if args.data_fetch:
        crypto.make_dataset(args.symbol,args.months,args.window_days,args.resample_hours,args.horizon,args.step)
    if args.train:
        crypto.TrainAll(f"{args.symbol}_{args.window_days}_{args.resample_hours}_{args.horizon}")
