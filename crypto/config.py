import os
##from dotenv import load_dotenv somehow loadenv not working



PRIVATE_KEY = None
API_KEY = None
RPC_ENDPOINT = None
MAX_SLIPPAGE = None
JUPITER_API_KEY = None
WALLET_DATA_DIR = None
LSTM_DATA_DIR = None


def load_config():
    env_path = '.env'
    global PRIVATE_KEY, RPC_ENDPOINT, API_KEY, MAX_SLIPPAGE, JUPITER_API_KEY, DEXSCREENER_API_KEY, DEXSCREENER_BASE_URL, SOL_TOKEN_ADDRESS, WALLET_DATA_DIR, LSTM_DATA_DIR



    config = {}
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()

                    if not line or line.startswith('#'):
                        continue

                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()

                        if (value.startswith('"') and value.endswith('"')) or (
                                value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        config[key] = value

        except Exception as e:
            print(f"Error reading .env file: {e}")
            exit()
    else:
        print(f"WARNING: .env file not found at {os.path.abspath(env_path)}")
        # exit()

    API_KEY = config.get("API_KEY")

    if not API_KEY:
        print("WARNING: API_KEY is required but not found in .env file!")
        print("Available keys in .env:", list(config.keys()))
        # exit()

    PRIVATE_KEY = config.get("PRIVATE_KEY")
    RPC_ENDPOINT = config.get("RPC_ENDPOINT", "https://api.mainnet-beta.solana.com")
    MAX_SLIPPAGE = int(config.get("MAX_SLIPPAGE", 50))
    JUPITER_API_KEY = config.get("JUPITER_API_KEY")

    WALLET_DATA_DIR = config.get("WALLET_DATA_DIR", "/home/wallet_data")
    LSTM_DATA_DIR = config.get("LSTM_DATA_DIR", "/home/lstm_data")

