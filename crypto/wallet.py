import time

from solders.pubkey import Pubkey
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solana.rpc.api import Client
from solana.rpc.types import TxOpts

import json
import requests

from .config import *

client = None
keypair = None



def get_token_balance(pubkey,token_address):
    owner = str(pubkey)
    mint = str(token_address)
    resp = client.get_token_accounts_by_owner(owner, {"mint": str(mint)})
    if not resp.value:
        return 0

    token_account_pubkey = Pubkey.from_string(resp.value[0]["pubkey"])
    balance_info = client.get_token_account_balance(token_account_pubkey)
    return int(balance_info.value.amount)
def get_best_quote(input_mint, output_mint, amount):

    try:
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": amount,
            "slippageBps": MAX_SLIPPAGE,
            "onlyDirectRoutes": True
        }
        response = requests.get("https://lite-api.jup.ag/swap/v1/quote", params=params)
        response.raise_for_status()
        quotes = response.json()
        return quotes.get("data", [])[0] if quotes.get("data") else None
    except Exception :
        return None
def create_swap_transaction(quote):
    try:
        params = {
            "userPublicKey": str(keypair.pubkey()),
            "quoteResponse": quote,
            "wrapAndUnwrapSol": True,
            "useSharedAccounts": True,
            "dynamicComputeUnitLimit": True,
        }
        response = requests.post(
            f"https://lite-api.jup.ag/swap/v1/swap",
            headers={"Content-Type": "application/json"},
            data=json.dumps(params),
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None

def buy(sol_ammount,output_token:str):
    best_route = get_best_quote(SOL_TOKEN_ADDRESS,output_token,sol_ammount * 1e9)
    if not best_route:
        return -1
    swap_txn = create_swap_transaction(best_route)
    if not swap_txn:
        return -1
    swap_transaction = VersionedTransaction.from_bytes(bytes(swap_txn["swapTransaction"]))
    payer = Keypair.from_base58_string(PRIVATE_KEY)
    swap_transaction.sign([payer])
    _ = client.send_transaction(swap_transaction,
        opts=TxOpts(skip_preflight=True)
    )["result"]
    return 0
def sell(token_address):
   balance = get_token_balance(keypair.public_key(),token_address)
   best_route = get_best_quote(token_address, SOL_TOKEN_ADDRESS, balance)
   if not best_route:
       return -1
   swap_txn = create_swap_transaction(best_route)
   if not swap_txn:
       return -1
   swap_transaction = VersionedTransaction.from_bytes(bytes(swap_txn["swapTransaction"]))
   payer = Keypair.from_base58_string(PRIVATE_KEY)
   swap_transaction.sign([payer])
   _ = client.send_transaction(swap_transaction,
       opts=TxOpts(skip_preflight=True)
   )["result"]
   return 0






def connect():
    global client, keypair

    client = Client(RPC_ENDPOINT)

    keypair = Keypair.from_base58_string(PRIVATE_KEY)
    balance = client.get_balance(keypair.public_key())
    print(f"SOL Balance: {balance['result']['value'] / 1e9} SOL")