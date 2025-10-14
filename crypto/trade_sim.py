import threading
import time
from datetime import timedelta
from datetime import datetime
import hashlib

from .data_fetch import *


class Trading_Sim:
    def __init__(self, initial_balance_usd,
                 buy_threshold=2, stop_loss=0.20, default_buy_amount=0.01, take_profit=2, runtime=6):
        self.initial_balance_usd = initial_balance_usd
        self.balance = initial_balance_usd
        self.buy_threshold = buy_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.default_buy_amount = default_buy_amount
        self.runtime = runtime
        self.wins = 0
        self.trades = 0
        self.losses = 0
        self.trades_lock = threading.Lock()
        self.stop_running = threading.Event()

    def model_trading_test(self):  ##trading using the LSTM model
        return

    def trade_test(self, token_address):
        time.sleep(5)  ##assume we buy a little bit later
        trade_amount = (self.initial_balance_usd * self.default_buy_amount)
        if self.balance < (self.initial_balance_usd * self.default_buy_amount):

            return
        token_security = get_token_security(token_address)
        if token_security is None:
            return
        if safe_float(token_security["top_10_holder_rate"]) > 0.25:
            return
        if token_security["is_show_alert"] is True:
            return
        if token_security["renounced_mint"] is not True:
            return
        if token_security["hide_risk"] is True:
            return
        buy_price = get_token_price(token=token_address)
        if buy_price == None:
            print("error getting price")
            return
        token_amount = trade_amount / buy_price
        while not self.stop_running.is_set():
            start_time = time.perf_counter()
            current_price = get_token_price(token=token_address)
            if current_price != None:
                if current_price / buy_price < (1 - self.stop_loss):
                    with self.trades_lock:
                        self.balance -= (self.initial_balance_usd * self.default_buy_amount)
                        self.balance += (token_amount * current_price)
                        self.trades += 1
                        self.losses += 1
                        print(f"stop_loss for token:{token_address}")
                        return
                if current_price / buy_price > self.take_profit:
                    with self.trades_lock:
                        self.balance -= (self.initial_balance_usd * self.default_buy_amount)
                        self.balance += (token_amount * current_price)
                        self.trades += 1
                        self.wins += 1
                        print(f"take profit for token:{token_address}")
                        return
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, float(60) - elapsed)
            time.sleep(sleep_time)

        with self.trades_lock:
            current_price = get_token_price(token=token_address)
            self.balance -= (self.initial_balance_usd * self.default_buy_amount)
            self.balance += (token_amount * current_price)
            self.trades += 1

    def copytrading_test(self):
        processed_tx_hashes = set()
        tokens_bought = {}
        tokens = {}
        wallets = json_get_merged_wallets()
        end_time = datetime.now() + timedelta(hours=self.runtime)
        self.stop_running.clear()
        while datetime.now() < end_time and self.balance > 10:
            time.sleep(5)
            if len(processed_tx_hashes) > 10000:
                processed_tx_hashes = set(list(processed_tx_hashes)[-10000:])

            for wallet in wallets:
                wallet_trades = get_wallet_buys(wallet=wallet)
                if wallet_trades is not None:
                    for trade in wallet_trades:
                        bought = trade["token_address"]
                        combination = f"{bought}_{wallet}"
                        hash = hashlib.sha256(combination.encode()).hexdigest()
                        if hash in processed_tx_hashes:
                            continue

                        processed_tx_hashes.add(hash)
                        if tokens_bought.get(bought, 0) != 1:
                            tokens[bought] = tokens.get(bought, 0) + 1
                            if tokens[bought] == self.buy_threshold:

                                trade_thread = threading.Thread(target=self.trade_test, args=(bought,))
                                trade_thread.start()
                                tokens_bought[bought] = 1

        self.stop_running.set()
        print("-" * 80)

        print("\nCopytrading backtest Results:")
        print(f"Treshold:{self.buy_threshold}")
        print(f"Runtime:{self.runtime} hours")
        print(f"Default buy ammount:{self.default_buy_amount * 100}%")
        print(f"Stop loss:{self.stop_loss} ")
        print(f"Take_profit:{self.take_profit}")
        print(f"Initial Balance: {self.initial_balance_usd:.2f}$")
        print(f"Final Balance: {self.balance:.2f}$")
        print(f"Profit/Loss: {self.balance - self.initial_balance_usd:.2f}$")
        print(f"Total Trades: {self.trades}")
        print(f"Wins: {self.wins}")
        if self.trades > 0:
            print(f"Win Rate: {(self.wins / self.trades) * 100:.2f}%")

        print("-" * 80)






