from .wallet import *
from .data_fetch import (
    json_get_merged_wallets,
    get_wallet_buys
)
from .config import *
from datetime import datetime, timedelta


def copytrade(treshold=2,runtime=6):

    tokens_bought = {}
    tokens = {}
    wallets = json_get_merged_wallets()
    end_time = datetime.now() + timedelta(hours=runtime)
    while datetime.now() < end_time:
        for wallet in wallets:
           wallet_trades = get_wallet_buys(wallet=wallet)
           if wallet_trades is not None:
            for bought in wallet_trades:
                if tokens_bought.get(bought,0) != 1:
                    tokens[bought] = tokens.get(bought,0) +1
                    if tokens[bought] == treshold:
                        tokens_bought[bought] = 1




def get_trade_ammount(token):

    balance = client.get_balance(keypair.public_key())
    trade_ammount = (balance/1e9) * 0.05
    return trade_ammount


