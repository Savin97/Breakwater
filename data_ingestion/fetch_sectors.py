#fetch_sectors.py
import pandas as pd
from pathlib import Path
import yfinance as yf



from config import TICKERS_FILE_PATH
from data_utilities.clean_input import read_tickers_to_fetch

def fetch_single_sector(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        raise LookupError("Fetching failed") from e

def fetch_sectors():
    tickers = read_tickers_to_fetch(Path(TICKERS_FILE_PATH))
    sectors = []
    for ticker in tickers:
        sector = fetch_single_sector(ticker)
        sectors.append(sector)

    return sectors