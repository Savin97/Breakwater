#fetch_sectors.py
import pandas as pd
from pathlib import Path
import yfinance as yf



from config import TICKERS_FILE_PATH
from data_utilities.clean_input import read_tickers_to_fetch

def fetch_single_sector(ticker: str) -> tuple:
    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector")
        sub_sector = info.get("industryDisp")
    except Exception as e:
        raise ValueError(f"Fetching Sector Name failed for {ticker}") from e
    return sector, sub_sector

def fetch_sectors() -> pd.DataFrame:
    tickers = read_tickers_to_fetch(Path(TICKERS_FILE_PATH))
    sector_dict = {}
    sub_sector_dict = {}
    for ticker in tickers:
        sector, sub_sector = fetch_single_sector(ticker)
        sector_dict[ticker] = sector
        sub_sector_dict[ticker] = sub_sector
    sector_df = pd.DataFrame({
        "sector": sector_dict,
        "sub_sector": sub_sector_dict
    })
    return sector_df