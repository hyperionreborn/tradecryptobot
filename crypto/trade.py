from .wallet import *
from .data_fetch import (
    json_get_merged_wallets,
    get_wallet_buys
)
from .config import *
from datetime import datetime, timedelta







def get_trade_ammount(token):

    balance = client.get_balance(keypair.public_key())
    trade_ammount = (balance/1e9) * 0.05
    return trade_ammount


