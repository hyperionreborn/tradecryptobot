import requests
import time
from datetime import datetime
import numpy as np
import os
import json
import asyncio
import aiohttp
from typing import List, Optional, Dict, Any
import yfinance as yf

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


def download_data(symbol,months,interval="1h"):
    if months == -1:
        df = yf.download(symbol,period="max",interval=interval,progress=True)
    else:
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=months*30)
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=True)
    df = df.sort_index()
    return df
def







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




if __name__ == "__main__":
    for _ in range(5):  ##100 token data
        _, _ = data_get(change=1, hours_collect=1)