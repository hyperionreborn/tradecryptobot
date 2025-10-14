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
    parser.add_argument("--change", type=float, default=1, help="number of hours to wait for price change")
    parser.add_argument("--hours_collect", type=int, default=1, help="number of hours data was collected")
    parser.add_argument("--get_wallets",action="store_true",help="get good wallets")
    parser.add_argument("--backtest",action="store_true",help="backtesting")
    parser.add_argument("--treshold",type=int,default=2,help="copytrading buy treshold")
    parser.add_argument("--stop_loss",type=float,default=0.35,help="stop loss")
    parser.add_argument("--initial_balance",type=int,default=1000,help="initial balance for backtesting")
    parser.add_argument("--default_buy_ammount",type=float,default=0.01,help="default buy ammount")
    parser.add_argument("--take_profit",type=float,default=2,help="take profit")
    parser.add_argument("--test_nlp",action="store_true",help="test the nlp model")
    parser.add_argument("--runtime",type=float,default=6,help="runtime")


    args = parser.parse_args()
    crypto.load_config()
    ##.connect()
    crypto.api_up()
    if args.data_fetch:
        for i in range(48):
            _, _ = crypto.data_get(
                change=int(args.change), hours_collect=int(args.hours_collect)
            )
    if args.get_wallets:
        asyncio.run(crypto.get_wallets())
    if args.backtest:

            copytrading_sim = crypto.Trading_Sim(
                initial_balance_usd=args.initial_balance,
                buy_threshold=args.treshold,
                stop_loss=args.stop_loss,
                default_buy_amount=args.default_buy_ammount,
                take_profit=args.take_profit,
                runtime=args.runtime
            )
            copytrading_sim.copytrading_test()


    if args.train:
        crypto.TrainAll(hours_collect=int(args.hours_collect))

    if args.predict:
        for i in range(4):
            crypto.Predict(
                binarymodel_path=f"{args.hours_collect}_{args.change}lstm_model.pt",
                pricemodel_path=f"{args.hours_collect}_{args.change}price_model.pt",
                collect_time=args.hours_collect,
            )
    if args.test_nlp:
        crypto.test_nlp()
    if args.test:
        results = crypto.test_predictions(args.test)
        print("\nPrediction Results:")
        print("-" * 80)
        for r in results:
            print(f"Token: {r['token']}")
            print(f"Prediction: {r['prediction']} | Prob: {r['probability']:.2%}")
            print(
                f"Binary Price: ${r['binary_price']:.2f} | Model Price: ${r['model_price']:.2f}"
            )
            print("-" * 80)
