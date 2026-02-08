import json
import time
from pathlib import Path

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
    parser.add_argument("--months", type=int, default=10, help="Number of past months to fetch (-1 for full history)")
    parser.add_argument("--step", type=int, default=1, help="Stride between sliding windows")
    parser.add_argument("--outdir", type=str, default="./dataset", help="Output directory for dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--backtest",action="store_true",help="backtesting")
    parser.add_argument("--test_nlp",action="store_true",help="test the nlp model")
    parser.add_argument("--symbol",type=str,default="BTCUSDT",help="pair to train on")
    parser.add_argument("--ensemble_train", action="store_true", help="train an ensemble model")
    parser.add_argument("--ensemble_eval", action="store_true", help="evaluate ensemble model")
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
                )
                time.sleep(1)
    if args.ensemble_eval:
        if args.symbol!="all":
            crypto.Ensemble_model_evaluate(f"{args.symbol}_{args.window_days}_{args.resample_hours}_{args.horizon}")
        else:
            for symbol in symbols:
                crypto.Ensemble_model_evaluate(f"{symbol}_{args.window_days}_{args.resample_hours}_{args.horizon}")
    if args.ensemble_train:
        if args.symbol!="all":
        # Prefer new dataset training if available
            crypto.Ensemble_model_train(f"{args.symbol}_{args.window_days}_{args.resample_hours}_{args.horizon}")
        # Fallback or alternative: crypto.TrainAll(hours_collect=int(args.hours_collect))
        else:
            for symbol in symbols:
                crypto.Ensemble_model_train(f"{symbol}_{args.window_days}_{args.resample_hours}_{args.horizon}")

    if args.train:
        if args.symbol!="all":
        # Prefer new dataset training if available
            acc,conf_acc = crypto.TrainAll(dataset_dir=f"{args.symbol}_{args.window_days}_{args.resample_hours}_{args.horizon}",EPOCHS=args.epochs)
        # Fallback or alternative: crypto.TrainAll(hours_collect=int(args.hours_collect))
        else:
            ar_acc = []
            arr_conf = []
            for symbol in symbols:
                acc,conf_acc = crypto.TrainAll(dataset_dir=f"{symbol}_{args.window_days}_{args.resample_hours}_{args.horizon}",
                                EPOCHS=args.epochs)
                ar_acc.append(acc)
                arr_conf.append(conf_acc)
            model_info = {
                'symbols':symbols,
                'accuracies':ar_acc,
                'confidence_accuracies':conf_acc
            }
            modelinfo_path =  "model_info.json"
            with open(modelinfo_path, 'w') as f:
                json.dump(model_info, f, indent=2)

    if args.backtest:
        crypto.test_live(args.symbol,args.resample_hours,args.window_days,args.horizon)
    if args.test_nlp:
        crypto.test_nlp()

