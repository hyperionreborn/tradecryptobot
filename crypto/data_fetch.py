import requests
import time
from datetime import datetime
import numpy as np
import os
import json
import asyncio
import aiohttp
from typing import List, Optional, Dict, Any

from numpy.ma.core import negative

from .config import *
api_semaphore = asyncio.Semaphore(4)
filter_semaphore = asyncio.Semaphore(4)
def api_up():
    url = f"{DEXSCREENER_BASE_URL}/health"
    response = requests.get(url)
    if response.status_code !=200:
        print("API is not working , exiting")
        exit()

async def fetch_json(session: aiohttp.ClientSession, url: str, max_retries: int = 3) -> Optional[Dict]:
    await asyncio.sleep(1)
    for retry in range(max_retries):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"HTTP {response.status} for {url}")
                    await asyncio.sleep(1)
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            await asyncio.sleep(1)
    return None

async def filter_wallets(session: aiohttp.ClientSession,wallets, api_key=None):
    if api_key is None:
        api_key = DEXSCREENER_API_KEY
    filtered_wallets = []
    urls = [f"{DEXSCREENER_BASE_URL}/wallet_stats/{wallet}?api_key={api_key}" for wallet in wallets]
    print("filtering wallets")
    await asyncio.sleep(1)
    
    async with filter_semaphore:
        tasks = []
        for url in urls:
            # Create task for each request - they start immediately!
            task = asyncio.create_task(fetch_json(session, url))
            tasks.append(task)
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process responses
        for wallet, data in zip(wallets, responses):
            if isinstance(data, Exception):
                print(f"Error processing {wallet}: {str(data)}")
                continue

            if data and isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
                stats = data["data"]
                winrate = stats.get("winrate", 0)
                buy_1d = stats.get("buy_1d", 0)
                avg_holding_period = stats.get("avg_holding_peroid", 0)
                pnl_7d = stats.get("pnl_7d", 0)
                honeypot_ratio = stats.get("risk", {}).get("token_honeypot_ratio", 1)
                sell_pass_buy_ratio = stats.get("risk",{}).get("sell_pass_buy_ratio",0)
                negative_trades = (stats.get("pnl_lt_minus_dot5_num",0)*0.5+stats.get("pnl_minus_dot5_0x_num",0)*1)
                positive_trades = (stats.get("pnl_lt_2x_num",0)*0.5+stats.get("pnl_2x_5x_num",0)*1 + stats.get("pnl_gt_5x_num",0)*5)
                roi = safe_divide(positive_trades,negative_trades,1.1)


                if (
                        winrate > 0.5 and winrate < 0.80 and
                        roi > 1 and
                        buy_1d < 80 and
                        avg_holding_period > 7200  and
                        pnl_7d > 0 and
                        honeypot_ratio == 0 and
                        sell_pass_buy_ratio < 0.1


                ):
                    filtered_wallets.append(wallet)
                    print(f"PASSED: {wallet}")
                    print(f"Winrate : {winrate * 100}%")
                    print(f"Buy 1d :  {buy_1d}")
                    print(f"Buy 1d: {buy_1d}")
                    print(
                        f"Avg Holding Period: {avg_holding_period} seconds ({avg_holding_period / 3600:.1f} hours)")
                    print(f"PNL 7d: {pnl_7d:.2f}")
                    print(f"Honeypot Ratio: {honeypot_ratio}")

        return filtered_wallets

async def get_wallets_for_token(session: aiohttp.ClientSession,token,limit=100,api_key=None,save_dir=None):
    if api_key is None:
        api_key = DEXSCREENER_API_KEY
    if save_dir is None:
        save_dir = WALLET_DATA_DIR

    print(f"Getting wallets for token: {token}")

    url = f"{DEXSCREENER_BASE_URL}/traders_top/{token}?limit={limit}&type=realized_profit&api_key={api_key}"

    async with api_semaphore:
        fetch_task = asyncio.create_task(fetch_json(session, url))
        data = await fetch_task

        if not data:
            return None

        if (data and isinstance(data, dict) and "data" in data and
                isinstance(data["data"], dict) and "list" in data["data"] and
                isinstance(data["data"]["list"], list)):
            traders = data["data"]["list"]
            addresses = [
                trader.get("address") for trader in traders
                if (isinstance(trader, dict) and "address" in trader and
                    trader.get("is_suspicious") is False)
            ]

            addresses = await filter_wallets(session, addresses)
            print(f"Found {len(addresses)} good wallets for token {token}")


            filename = os.path.join(save_dir, f"{token}.json")
            with open(filename, "w") as out:
                json.dump(addresses, out, indent=2)
            return addresses


    return None


def get_wallet_buys(wallet, api_key=None, limit=100):
    if api_key is None:
        api_key = DEXSCREENER_API_KEY
    url = f"{DEXSCREENER_BASE_URL}/wallet_trades/{wallet}?lim={limit}&type=buy&api_key={api_key}"
    retries = 0
    while retries < 3:
        try:
            response = requests.get(url)
            if response.status_code != 200:
                print(f"HTTP {response.status_code} \n ")
                retries += 1
                time.sleep(1)
                continue
            data = response.json()

            if (
                    data
                    and isinstance(data, dict)
                    and "data" in data
                    and isinstance(data["data"], dict)
                    and "activities" in data["data"]
                    and isinstance(data["data"]["activities"], list)
            ):
                activities = data["data"]["activities"]

                # Create a list of dictionaries with token address and tx_hash
                wallet_buys = []
                seen_tokens = set()  # To maintain uniqueness by token address

                for activity in activities:
                    token_address = activity.get("token", {}).get("address")
                    tx_hash = activity.get("tx_hash")

                    if token_address is not None and tx_hash is not None:
                        # Only add if we haven't seen this token address yet
                        if token_address not in seen_tokens:
                            wallet_buys.append({
                                "token_address": token_address,
                                "tx_hash": tx_hash,
                                "timestamp": activity.get("timestamp"),
                                "symbol": activity.get("token", {}).get("symbol")
                            })
                            seen_tokens.add(token_address)

                return wallet_buys
            else:
                retries += 1
                time.sleep(1)
                continue
        except Exception as e:
            print(f"Error: {str(e)} \n")
            time.sleep(1)
            retries += 1
            continue

    return None


def get_trending_tokens(trending_time="6h",limit=50,api_key=None):
        if api_key is None:
            api_key = DEXSCREENER_API_KEY
        print("getting trending tokens")
        url = f"{DEXSCREENER_BASE_URL}/trending/pairs?hours={trending_time}&api_key={api_key}"
        retries =0
        while retries < 3:
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    print(f"HTTP {response.status_code} \n")
                    time.sleep(2)
                    retries +=1
                    continue
                data = response.json()

                if (
                        data
                        and isinstance(data, dict)
                        and "data" in data
                        and isinstance(data["data"], dict)
                        and "rank" in data["data"]
                        and isinstance(data["data"]["rank"], list)
                ):
                    tokens = data["data"]["rank"]
                    addresses = [token.get("address") for token in tokens if isinstance(token,dict) and token.get("address") is not None]
                    return addresses[:limit]
            except Exception as e:
                print(f"Error : {str(e)} \n")
                time.sleep(2)
                retries +=1
                continue

        return None




async def get_wallets(limit=100,trending_time="6h",save_dir=None):
    if save_dir is None:
        save_dir = WALLET_DATA_DIR
    if os.path.exists(save_dir):
        for filename in os.listdir(save_dir):
            os.remove(os.path.join(save_dir,filename)) ##remove the old wallet list
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    tokens = get_trending_tokens(trending_time=trending_time)
    print(f"found {len(tokens)} tokens")
    connector = aiohttp.TCPConnector(limit=10, limit_per_host=10)
    async with aiohttp.ClientSession(connector=connector) as session:

        tasks = []
        for token in tokens:
            task = asyncio.create_task(get_wallets_for_token(session=session,token=token,limit=limit))
            tasks.append(task)

        responses = await asyncio.gather(*tasks,return_exceptions=True)
        for response in responses:
            if isinstance(response, Exception):
                print(f"Token processing error: {response}")
    wallet_list = json_get_merged_wallets(dir=save_dir)
    print(f"got {len(wallet_list)} wallets")
    return wallet_list
def get_token_age(pair_created_at):
    if not pair_created_at:
        return None
    created_time = datetime.fromtimestamp(pair_created_at / 1000)
    now = datetime.now()
    age = now - created_time
    return age.total_seconds() / 3600

def safe_divide(numerator, denominator, default=1.0):

    try:
        num = float(numerator)
        den = float(denominator)
        if den <= 0:
            return default
        return num / den
    except (TypeError, ValueError, ZeroDivisionError):
        return default
def safe_float(value):
    """Helper function to safely convert values to float"""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        # For nested dictionaries, try to get the first numeric value
        for v in value.values():
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0
def get_token_price(token,api_key=None):
    if api_key is None:
        api_key = DEXSCREENER_API_KEY
    url = f"{DEXSCREENER_BASE_URL}/pair/{token}?api_key={api_key}"
    retries = 0
    while retries < 3:
        try:
            response = requests.get(url)
            if response.status_code != 200:
                print(f"HTTP {response.status_code} for token {token} \n")
                return None

            data = response.json()
            if not data:
                print("error in getting response \n")
                retries +=1
                time.sleep(1)
                continue
            data_array = data.get("data", [])
            token_data = data_array[0]

            if token != token_data.get("address"):
                print("error in retrieving right token info \n")

            try:
                price_data = token_data.get("price")


                price =  safe_float(price_data.get("price_1m"))



                return price

            except Exception as e:
                print(f"Error processing pair data for token {token}: {str(e)} \n")
                retries +=1
                time.sleep(1)
                continue

        except Exception as e:
            print(f"Error fetching data for token {token}: {str(e)} \n")
            retries +=1
            time.sleep(1)
            continue

    return None

def get_token_data(token,api_key=None):
        if api_key is None:
            api_key = DEXSCREENER_API_KEY
        url = f"{DEXSCREENER_BASE_URL}/pair/{token}?api_key={api_key}"
        retries = 0
        while retries < 3:
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    print(f"HTTP {response.status_code} for token {token} \n")
                    retries += 1
                    time.sleep(1)
                    continue

                data = response.json()
                if not data:
                    print("error in getting response \n")
                    retries +=1
                    time.sleep(1)
                    continue
                data_array = data.get("data", [])
                token_data = data_array[0]



                if token != token_data.get("address"):
                    print("error in retrieving right token info \n")
                    retries += 1
                    time.sleep(1)
                    continue


                try:
                    price_data = token_data.get("price")


                    processed_data = {
                        "sells1m": safe_float(price_data.get("sells_1m")),
                        "buys1m": safe_float(price_data.get("buys_1m")),
                        "swaps1m": safe_float(price_data.get("swaps_1m")),
                        "age": safe_float(get_token_age(token_data.get("creation_timestamp"))),
                        "volume": safe_float(price_data.get("volume_1m")),
                        "price1m": safe_float(
                            price_data.get("price_1m")
                        ),
                        "buy_sell_ratio": safe_divide(safe_float(price_data.get("buy_volume_1m")),safe_float(price_data.get("sell_volume_1m"))),
                        "price5m": safe_float(price_data.get("price_5m")),
                        "liquidity": safe_float(price_data.get("liquidity")),
                        "holders": safe_float(token_data.get("holder_count")),
                        "top10holders": safe_float(
                            token_data.get("dev",{}).get("top_10_holder_rate")
                        )

                    }

                    # Validate all values are proper floats
                    for key, value in processed_data.items():
                        if not isinstance(value, float):
                            print(
                                f"Warning: Non-float value for {key}: {value} in token {token} \n"
                            )
                            processed_data[key] = 0.0

                    return processed_data

                except Exception as e:
                    print(f"Error processing pair data for token {token}: {str(e)} \n")
                    return None

            except Exception as e:
                print(f"Error fetching data for token {token}: {str(e)} \n")
                return None

        return None


def get_token_security(token):
    url = f"{DEXSCREENER_BASE_URL}/token_sec/{token}"
    retries = 0
    while retries < 3:
        try:
            response = requests.get(url)
            if response.status_code != 200:
                print(f"HTTP {response.status_code} for token {token} \n")
                retries +=1
                time.sleep(1)
                continue
            data = response.json()

            if (
                    data
                    and isinstance(data, dict)
                    and "data" in data
                    and isinstance(data["data"], dict)
            ):
                token_security = data["data"]
                return token_security
            else:
                retries += 1
                time.sleep(1)
                continue
        except Exception as e:
            print(f"Error fetching security data for token {token}: {str(e)} \n")
            retries += 1
            time.sleep(1)
            continue
    return None

def get_new_pairs(limit=50):
    url = f"{DEXSCREENER_BASE_URL}/new/pairs?limit={limit}"

    try:

        response = requests.get(url)

        if response.status_code != 200:
            print(f"HTTP {response.status_code} when taking snapshot, retrying... \n")
            return None

        data = response.json()
        if not data:
            print("error in getting response \n")
            return None

        if (
                data
                and isinstance(data, dict)
                and "data" in data
                and isinstance(data["data"], dict)
                and "pairs" in data["data"]
                and isinstance(data["data"]["pairs"], list)
        ):
            pairs = data.get("data", {}).get("pairs", [])
            tokens = [pair["base_address"] for pair in pairs if
                      (isinstance(pair, dict) and "base_address" in pair and "quote_address" in pair)]
            return tokens

    except Exception:
        return None
def take_snapshot( type,limit=50,trending_time = "6h"):

    tokens = None
    if type == "new":
       tokens = get_new_pairs(limit)
    if type == "trending":
       tokens =  get_trending_tokens(trending_time,limit)
    return tokens






def collect_data_for_tokens(tokens, interval=60, max_data_points=60):
    if not tokens:
        return None

    collected_data = {token: [] for token in tokens}
    failed_tokens = set()

    for i in range(max_data_points):
        start_time = time.perf_counter()
        print(f"Collecting datapoint {i + 1}/{max_data_points} \n")

        for token_address in tokens:
            if token_address in failed_tokens:
                continue

            token_address = token_address
            try:
                new_data = get_token_data(token_address)
                if new_data is not None:
                    collected_data[token_address].append(new_data)
                else:
                    print(f"Failed to get data for token {token_address} \n")
                    failed_tokens.add(token_address)
            except Exception as e:
                print(f"Error collecting data for token {token_address}: {str(e)} \n")
                failed_tokens.add(token_address)

        elapsed = time.perf_counter() - start_time
        sleep_time = max(0, float(interval) - elapsed)
        if sleep_time > 0:
            print(f"Waiting {sleep_time:.1f}s until next collection... \n")
            time.sleep(sleep_time)

    # Remove tokens with insufficient data
    for token_address in list(collected_data.keys()):
        if len(collected_data[token_address]) < max_data_points:
            print(f"Removing token {token_address} due to insufficient data points \n")
            del collected_data[token_address]

    return collected_data


def convert(
    collected_data,
    features=[
        "sells1m",
        "buys1m",
        "swaps1m",
        "age",
        "volume",
        "price1m",
        "buy_sell_ratio",
        "price5m",
        "liquidity",
        "holders",
        "top10holders",
    ],
):
    lstm_data = {}

    for token_address, raw_data in collected_data.items():
        try:
            # Validate data before conversion
            if not raw_data or not isinstance(raw_data, list):
                print(f"Invalid data format for token {token_address} \n")
                continue

            X = []
            for item in raw_data:
                try:
                    row = []
                    for feature in features:
                        value = item.get(feature, 0.0)
                        # Ensure we have a numeric value
                        row.append(float(value) if value is not None else 0.0)
                    X.append(row)
                except (ValueError, TypeError) as e:
                    print(f"Error processing row for token {token_address}: {str(e)} \n")
                    continue

            if len(X) > 0:
                X = np.array(X, dtype=np.float32)
                if X.shape[1] == len(features):
                    lstm_data[token_address] = X
                else:
                    print(f"Incorrect feature count for token {token_address} \n")

        except Exception as e:
            print(f"Error converting data for token {token_address}: {str(e)} \n")
            continue

    return lstm_data


def generate_labels(collected_data, threshold=0.0):
    labels = {}
    for token_address, data_points in collected_data.items():
        # if len(data_points) < 61:
        # continue  # not enough data to calculate label

        try:
            price_now = float(data_points[-2]["price1m"])
            price_later = float(data_points[-1]["price1m"])
            change = (price_later - price_now) / price_now

            labels[token_address] = np.array(int(change > threshold))
        except (TypeError, KeyError, ZeroDivisionError):
            continue

    return labels

def json_get_merged_wallets(dir=None):
    if dir is None:
        dir = WALLET_DATA_DIR
    merged = []

    if os.path.exists(os.path.join(dir, "merged_data")):
        os.remove(os.path.join(dir, "merged_data"))


        # Loop over all .json files
    for filename in os.listdir(dir):
        if filename.endswith(".json") :

            with open(os.path.join(dir, filename), "r") as file:
                try:
                    data = json.load(file)
                    if isinstance(data, list):  # wallets stored as array
                        merged.extend(data)

                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON file: {filename} \n")

    merged = list(dict.fromkeys(merged))

    with open(os.path.join(dir,"merged_data"), "w") as f:
        json.dump(merged, f)
    return merged
def json_get_merged_tokens(dir):
    merged = {}
    if not os.path.exists(dir):
        print(f"Directory '{dir}' does not exist. Please run data collection first. \n")
        return None

    if os.path.exists(os.path.join(dir, "merged_data")):
        os.remove(os.path.join(dir, "merged_data"))
    for filename in os.listdir(dir):
        if filename.endswith(".json"):
            with open(os.path.join(dir, filename), "r") as file:
                try:
                    data = json.load(file)
                    for token, data in data.items():
                        merged[token] = data
                except json.JSONDecodeError:
                    print(f"skipping invalid JSON file: {filename} \n")
    with open(os.path.join(dir, "merged_data"), "w") as f:
        json.dump(merged, f)
    return merged


def json_save(data, save_dir=None):
    if save_dir is None:
        save_dir = LSTM_DATA_DIR
    if not data:
        return
    os.makedirs(save_dir, exist_ok=True)  # Create the folder if it doesn't exist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"lstm_data_{timestamp}.json")

    lstm_data_json = {k: v.tolist() for k, v in data.items()}
    with open(filename, "w") as f:
        json.dump(lstm_data_json, f)


def json_get(filename):
    with open(filename, "r") as f:
        loaded_json_raw = json.load(f)
    data = {k: np.array(v, dtype=np.float32) for k, v in loaded_json_raw.items()}
    return data


def data_get(threshold=0.0, change=1, hours_collect=1,type="trending"):
    print("Starting data collection... \n")
    tokens = take_snapshot(limit=50,type=type)
    if not tokens:
        print("No tokens found. \n")
        return None, None

    print(f"Found {len(tokens)} tokens in snapshot \n")

    # Collect 60 time points (60 minutes)
    collected = collect_data_for_tokens(
        tokens, interval=60, max_data_points=int(60 * hours_collect)
    )
    if not collected:
        print("No data collected for any tokens \n")
        return None, None

    print(f"Collected historical data for {len(collected)} tokens \n")

    print(f"Waiting {change}h for target price... \n")
    time.sleep(int(3600 * change))

    # Add the 61st data point (change * 1h later)
    successful_labels = 0

    for token in list(collected.keys()):
        label_data = get_token_data(token)
        if label_data:
            collected[token].append(label_data)
            successful_labels += 1
        else:
            del collected[token]
    collection_time = datetime.now().strftime("%Y%m%d_%M%H")
    for token in list(collected.keys()):
        token_with_time = f"{collection_time}_{token}"
        collected[token_with_time] = collected[token]
        del collected[token]







    print(f"Collected target prices for {successful_labels} tokens \n")

    # Convert to LSTM format
    lstm_data = convert(collected)
    print(f"Converted data for {len(lstm_data)} tokens \n")
    labels = generate_labels(collected, threshold=threshold)
    # Generate binary labels
    print(f"Generated labels for {len(labels)} tokens \n")
    if len(lstm_data) == 0 or len(labels) == 0:
        print("No usable data after processing \n")
        return None, None
    json_save(lstm_data, f"lstm{hours_collect}h_{change}h")
    json_save(labels, f"lstm_labels{hours_collect}h_{change}h")
    print("Data saved to JSON files \n")
    return lstm_data, labels


if __name__ == "__main__":
    for _ in range(5):  ##100 token data
        _, _ = data_get(change=1, hours_collect=1)