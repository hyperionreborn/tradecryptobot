import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

import crypto
import argparse
import asyncio



if __name__ == "__main__":
    symbols = ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","LTCUSDT"]
    parser = argparse.ArgumentParser(description="Crypto prediction tool")
    parser.add_argument("--data_fetch", action="store_true", help="fetching data")
    parser.add_argument("--train", action="store_true", help="Training mode")
    parser.add_argument("--predict", action="store_true", help="Prediction mode")
    parser.add_argument("--test", type=str, help="Test predictions on saved data file")
    parser.add_argument("--window_days", type=int, default=28, help="Number of past days in each training window")
    parser.add_argument("--horizon", type=int, default=3, help="Number of resampled steps ahead to predict")
    parser.add_argument("--resample_hours", type=int, default=4, help="Resampling interval in hours")
    parser.add_argument("--months", type=int, default=8, help="Number of past months to fetch (-1 for full history)")
    parser.add_argument("--step", type=int, default=1, help="Stride between sliding windows")
    parser.add_argument("--outdir", type=str, default="./dataset", help="Output directory for dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--backtest",action="store_true",help="backtesting")
    parser.add_argument("--cutoff",type=int,default=0,help="how many days ago  should there be data fetch cutoff")
    parser.add_argument("--symbol",type=str,default="BTCUSDT",help="pair to train on")
    parser.add_argument("--grid_search",action="store_true",help="perform grid search for parameter tuning")
    args = parser.parse_args()
    crypto.load_config()
    ##.connect()
    # crypto.api_up()

    if args.data_fetch:
        if args.symbol!="all":
            crypto.make_dataset(
                symbol=args.symbol, # Defaulting to BTC-USD as per original intent or make it an arg? Adding symbol arg would be good too but sticking to requested ones first.
                months=args.months,
                window_days=args.window_days,
                resample_hours=args.resample_hours,
                horizon=args.horizon,
                step=args.step,
                cutoff=args.cutoff
            )
        else:
            for symbol in symbols:
                crypto.make_dataset(
                    symbol=symbol,
                    # Defaulting to BTC-USD as per original intent or make it an arg? Adding symbol arg would be good too but sticking to requested ones first.
                    months=args.months,
                    window_days=args.window_days,
                    resample_hours=args.resample_hours,
                    horizon=args.horizon,
                    step=args.step,
                    cutoff=args.cutoff
                )
                time.sleep(1)
    if args.grid_search:
        if args.symbol != "all":
            crypto.grid_search(dataset_dir=f"{args.symbol}_{args.window_days}_{args.resample_hours}_{args.horizon}")
        else:
            for symbol in symbols:
                crypto.grid_search(dataset_dir=f"{symbol}_{args.window_days}_{args.resample_hours}_{args.horizon}")



    if args.train:
        if args.symbol!="all":
        # Prefer new dataset training if available
            acc,conf_acc,_,_ = crypto.TrainAll(dataset_dir=f"{args.symbol}_{args.window_days}_{args.resample_hours}_{args.horizon}",EPOCHS=args.epochs)
        # Fallback or alternative: crypto.TrainAll(hours_collect=int(args.hours_collect))
        else:
            ar_acc = []
            arr_conf = []
            for symbol in symbols:
                acc,conf_acc,_,_ = crypto.TrainAll(dataset_dir=f"{symbol}_{args.window_days}_{args.resample_hours}_{args.horizon}",
                                EPOCHS=args.epochs)
                ar_acc.append(acc)
                arr_conf.append(conf_acc)
            model_info = {
                'symbols':symbols,
                'accuracies':ar_acc,
                'confidence_accuracies':conf_acc,
            }
            modelinfo_path =  "model_info.json"
            with open(modelinfo_path, 'w') as f:
                json.dump(model_info, f, indent=2)
    if args.backtest:
        i = 0
        stats = defaultdict(lambda: {'test_acc': [], 'test_conf': [], 'val_acc': [],'val_conf3.5d': []})
        while i < 61:
            ar_acc = []
            arr_conf = []
            for symbol in symbols:
                crypto.make_dataset(
                    symbol=symbol,
                        # Defaulting to BTC-USD as per original intent or make it an arg? Adding symbol arg would be good too but sticking to requested ones first.
                    months=args.months,
                    window_days=args.window_days,
                    resample_hours=args.resample_hours,
                    horizon=args.horizon,
                    step=args.step,
                    cutoff=i
                )
                time.sleep(1)

            for symbol in symbols:
                acc, conf_acc,val_acc,val_conf3_acc = crypto.TrainAll(
                    dataset_dir=f"{symbol}_{args.window_days}_{args.resample_hours}_{args.horizon}",
                    EPOCHS=args.epochs)
                stats[symbol]['test_acc'].append(acc)
                stats[symbol]['test_conf'].append(conf_acc)
                stats[symbol]['val_acc'].append(val_acc)
                stats[symbol]['val_conf3.5d'].append(val_conf3_acc)

            i += 3
        for symbol, s in stats.items():
            print(f"{symbol}:")
            print(f"  mean test acc  : {np.mean(s['test_acc']):.2%}")
            print(f"  mean test conf : {np.mean(s['test_conf']):.2%}")
            print(f"  mean val 7d acc   : {np.mean(s['val_acc']):.2%}")
            print(f"  min val 7d acc    : {np.min(s['val_acc']):.2%}")
            print(f"  std val 7d acc    : {np.std(s['val_acc']):.2%}")
            print(f"  mean val conf 3.5d acc   : {np.mean(s['val_conf3.5d']):.2%}")
            print(f"  min val conf 3.5d acc    : {np.min(s['val_conf3.5d']):.2%}")
            print(f"  std val conf 3.5d acc    : {np.std(s['val_conf3.5d']):.2%}")
    if args.backtest:
        crypto.test_live(args.symbol,args.resample_hours,args.window_days,args.horizon)

