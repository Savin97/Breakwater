# data_ingestion/api_functions.py
import requests
import time

from config import (ALPHAVANTAGE_BASE_URL, 
                    TIMEOUT_SECONDS,
                    BACKOFF_SECONDS)

from data_utilities.helper_funcs import get_alpha_vantage_api_key

def get_earnings_data_from_api(stock, output_size = "full"):
    api_key = get_alpha_vantage_api_key()
    params = {
        "function": "EARNINGS", 
        "symbol": stock, 
        "apikey": api_key,
        "outputsize": output_size}  
    r = requests.get(ALPHAVANTAGE_BASE_URL, params=params, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()
    data = r.json()       
    return data 

def handle_api_fetching_errors(stock, data):
    last_err = None
    # Rate-limit / errors
    if "Note" in data:
        last_err = RuntimeError(data["Note"])
        print(f"Alpha Vantage rate limit for {stock}: {data['Note']}")
        print(f"Waiting {BACKOFF_SECONDS} seconds")
        time.sleep(BACKOFF_SECONDS)
        # if attempt == MAX_RETRIES:
        #     raise RuntimeError(f"Alpha Vantage rate limit for {stock}: {data['Note']}")
    if "Error Message" in data:
        last_err = RuntimeError(data["Error Message"])
        raise RuntimeError(f"Alpha Vantage error for {stock}: {data['Error Message']}")
    if "Information" in data:
        last_err = RuntimeError(data["Information"])
        raise RuntimeError(f"Alpha Vantage error for {stock}: {data['Information']}")
    return last_err