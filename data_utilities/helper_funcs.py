import time
import os
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

from config import (BACKOFF_SECONDS, 
                    DEFAULT_REACTION_WINDOW,
                    USE_CACHED_DATA_FLAG,
                    STOCK_LIST_PATH)

def directory_checks():
    Path("data").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)

def read_stocks_to_fetch() -> list[str]:
    """
        Reads stocks from a file. Supports:
        - .txt: one stock per line
        - .csv: column named symbol/ticker/stock 
        Returns a list of all unique stocks (uppercase, no spaces)
    """
    path = Path(STOCK_LIST_PATH)

    if path.suffix.lower() == ".csv":
        print("Reading stocks from .csv file")
        stock_prices_df = pd.read_csv(path)
        col = None
        for c in ("stock", "symbol", "ticker","Stock", "Symbol", "Ticker"):
            if c in stock_prices_df.columns:
                col = c
                break
        if col is None:
            raise ValueError(f"CSV must contain a symbol/ticker/stock column. Found: {list(stock_prices_df.columns)}")
        stocks = stock_prices_df[col].astype(str).str.strip().tolist()
    else:
        print("Reading stocks from .txt file")
        stocks = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        print(stocks)

    # Basic cleanup
    stocks = [t.replace(" ", "").upper() for t in stocks if t]
    # dedupe preserve order
    seen = set()
    out = []
    for stock in stocks:
        if stock and stock != "NAN" and stock not in seen:
            out.append(stock)
            seen.add(stock)
    return out


def sleep_backoff(attempt: int) -> None:
    # exponential backoff with light jitter
    wait = BACKOFF_SECONDS * (2 ** attempt)
    wait = wait + (0.1 * wait)
    print(f"Waiting {wait} seconds")
    time.sleep(wait)

def get_alpha_vantage_api_key() -> str:
    load_dotenv() 
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing ALPHAVANTAGE_API_KEY environment variable"
        )
    return api_key


def chunk_list(items: list[str], n: int):
    """
        Takes a list and returns it in chunks of size n.
        Example:
        items = ["A", "B", "C", "D", "E", "F", "G"]; n = 3
        Output (one chunk at a time):
        ["A", "B", "C"]
        ["D", "E", "F"]
        ["G"]
        yield makes this a generator, not a normal function.
        That means:
            It does not return everything at once
            It returns one chunk at a time
            Memory-efficient
            Perfect for large lists (like hundreds of stocks)
    """
    for i in range(0, len(items), n):
        yield items[i:i+n]

def build_earnings_df(df):
    """ Separate earnings rows """

    # Boolean mask, gives True for rows with earnings dates
    earnings_mask =  df[DEFAULT_REACTION_WINDOW].notna()
    earnings_df = df.loc[earnings_mask]
    earnings_df = earnings_df.sort_values(["stock", "earnings_date"])
    return earnings_df


def check_cached_data_use():

    print(f"Cached Data Usage Switch is set to: {USE_CACHED_DATA_FLAG}\n")

    if USE_CACHED_DATA_FLAG == False:
        answer = input("USE_CACHED_DATA_FLAG is switched OFF - are you sure? Y/N ")
        if answer.lower() == "n":
            print("OK, Switch USE_CACHED_DATA_FLAG ON and retry.\nExecution Stopped.")
            exit()
        if answer.lower() == "y":
            print("OK, Model won't use Cached Data. Proceeding...\n")