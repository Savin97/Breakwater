#fetch_sectors.py
import pandas as pd
from pathlib import Path
import yfinance as yf

from config import STOCK_NAMES_FILE_PATH, SECTORS_PATH, USE_CACHED_DATA_FLAG
from data_utilities.clean_input import read_stocks_to_fetch

# def fetch_single_sector(stock: str) -> dict:
#     print(f"Fetching {stock} Sector Data\n")
#     try:
#         info = yf.Ticker(stock).info
#         sector = info.get("sector")
#         sub_sector = info.get("industryDisp") or info.get("industry")
#         market_cap = info.get('marketCap', None)
#         beta = info.get('beta', None)
#     except Exception as e:
#         row = {
#             "stock": stock,
#             "sector": None,
#             "sub_sector": None,
#             "market_cap": None,
#             "beta": None
#         }
#         print(f"[WARN] Failed fetching sector data for {stock}: {e}")
#         return row
#     row = {
#         "stock": stock,
#         "sector": sector,
#         "sub_sector": sub_sector,
#         "market_cap": market_cap,
#         "beta": beta
#     }
        
#     return row

# def fetch_sectors() -> pd.DataFrame:
#     """
#         Returns a DF
#         stock sector sub_sector
#     """
#     stocks = set(read_stocks_to_fetch(Path(STOCK_NAMES_FILE_PATH)))

#     if USE_CACHED_DATA_FLAG == True:
#         # Check cache
#         if Path(SECTORS_PATH).exists():
#             print("Sectors DF already exists in data/ , using CACHED Sectors")
#             cached_df = pd.read_csv(SECTORS_PATH)

#             # Stocks that aren't acturally complete
#             complete_mask = (
#                 cached_df["sector"].notna()
#                 & cached_df["sub_sector"].notna()
#                 & cached_df["market_cap"].notna()
#                 & cached_df["beta"].notna()
#             )

#             cached_stocks = set(cached_df.loc[complete_mask, "stock"])
#         else:
#             cached_df = pd.DataFrame(columns=["stock", "sector", "sub_sector","market_cap","beta"])
#             cached_stocks = set()
#         stocks_to_fetch = sorted(stocks - cached_stocks)
#     else:
#         stocks_to_fetch = sorted(stocks)
#         cached_df = pd.DataFrame()

#     rows = []

#     for stock in stocks_to_fetch:
#         row = fetch_single_sector(stock)
#         rows.append(row)

#     if rows:
#         new_stocks_df = pd.DataFrame(rows)
#         sector_df = pd.concat([cached_df, new_stocks_df], ignore_index=True)  
#         sector_df.to_csv(SECTORS_PATH, index = False)
#     else:
#         sector_df = cached_df
#     return sector_df




# Multi Threaded Version
import time
import random
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_single_sector(stock: str, retries: int = 2, base_sleep: float = 0.6) -> dict:
    print(f"Fetching {stock} Sector Data\n")

    for attempt in range(retries + 1):
        try:
            info = yf.Ticker(stock).info  # slow/flaky but OK for now
            sector = info.get("sector")
            sub_sector = info.get("industryDisp") or info.get("industry")
            market_cap = info.get("marketCap", None)
            beta = info.get("beta", None)

            return {
                "stock": stock,
                "sector": sector,
                "sub_sector": sub_sector,
                "market_cap": market_cap,
                "beta": beta
            }

        except Exception as e:
            if attempt < retries:
                # exponential backoff + jitter
                sleep_s = base_sleep * (2 ** attempt) + random.random() * 0.2
                time.sleep(sleep_s)
                continue

            print(f"[WARN] Failed fetching sector data for {stock}: {e}")
            return {
                "stock": stock,
                "sector": None,
                "sub_sector": None,
                "market_cap": None,
                "beta": None
            }

def fetch_sectors() -> pd.DataFrame:
    """
        Returns a DF
        stock sector sub_sector
    """
    stocks = set(read_stocks_to_fetch(Path(STOCK_NAMES_FILE_PATH)))

    if USE_CACHED_DATA_FLAG == True:
        # Check cache
        if Path(SECTORS_PATH).exists():
            print("Sectors DF already exists in data/ , using CACHED Sectors")
            cached_df = pd.read_csv(SECTORS_PATH)

            # Stocks that aren't acturally complete
            complete_mask = (
                cached_df["sector"].notna()
                & cached_df["sub_sector"].notna()
                & cached_df["market_cap"].notna()
                & cached_df["beta"].notna()
            )

            cached_stocks = set(cached_df.loc[complete_mask, "stock"])
        else:
            cached_df = pd.DataFrame(columns=["stock", "sector", "sub_sector","market_cap","beta"])
            cached_stocks = set()
        stocks_to_fetch = sorted(stocks - cached_stocks)
    else:
        stocks_to_fetch = sorted(stocks)
        cached_df = pd.DataFrame()

    rows = []

    max_workers = 5  # good default; bump to 12-16 if you have <300 tickers and it’s stable

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_single_sector, stock): stock for stock in stocks_to_fetch}

        for i, fut in enumerate(as_completed(futures), 1):
            stock = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as e:
                print(f"[WARN] Thread failed for {stock}: {e}")
                rows.append({
                    "stock": stock,
                    "sector": None,
                    "sub_sector": None,
                    "market_cap": None,
                    "beta": None
                })

            if i % 25 == 0 or i == len(stocks_to_fetch):
                print(f"Fetched {i}/{len(stocks_to_fetch)} sector rows")

    if rows:
        new_stocks_df = pd.DataFrame(rows)
        sector_df = pd.concat([cached_df, new_stocks_df], ignore_index=True)  
        sector_df = sector_df.drop_duplicates(subset=["stock"], keep="last")
        sector_df.to_csv(SECTORS_PATH, index = False)
    else:
        sector_df = cached_df
    return sector_df
