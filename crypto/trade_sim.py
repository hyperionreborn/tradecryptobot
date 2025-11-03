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



    

        self.stop_running.set()
        print("-" * 80)


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






