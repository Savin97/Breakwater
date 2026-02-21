"""
fetch_earnings.py

Fetch historical quarterly earnings *report dates* from Alpha Vantage.

- Provider: Alpha Vantage
- Output columns:
    stock, earnings_date, fiscal_date_ending

Usage:
    from data_ingestion.fetch_earnings import fetch_earnings_dates

    df = fetch_earnings_dates(["AAPL", "MSFT"], start_date=STOCKS_START_DATE)

Auth:
    env var that holds API key.
"""
import time
import requests
import pandas as pd
from typing import List
from pathlib import Path

from config import (STOCKS_START_DATE,
                    MAX_RETRIES,
                    BACKOFF_SECONDS,
                    EARNINGS_PATH,
                    STOCK_LIST_PATH)

from data_ingestion.api_functions import get_earnings_data_from_api, handle_api_fetching_errors
from data_utilities.formatting import parse_date
from data_utilities.helper_funcs import read_stocks_to_fetch, get_alpha_vantage_api_key


def fetch_earnings_dates_for_stock(stock: str) -> pd.DataFrame:
    """
        Fetch quarterly earnings report dates for a single stock from Alpha Vantage.
        Keeps only rows where reportedDate >= start_date.
    """
    last_err = None
    start_date = parse_date(STOCKS_START_DATE)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = get_earnings_data_from_api(stock)
        except Exception as exc:
            last_err = exc
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Alpha Vantage request failed for {stock}: {exc}")
            print(f"Waiting {BACKOFF_SECONDS * attempt} seconds")
            time.sleep(BACKOFF_SECONDS * attempt)
            continue
        
        last_err = handle_api_fetching_errors(stock, data)
        
        quarterly = data.get("quarterlyEarnings", []) or []
        rows = []
        for q in quarterly:
            reported = parse_date(q.get("reportedDate"))
            fiscal = parse_date(q.get("fiscalDateEnding"))
            if reported is None:
                continue
            if reported < start_date:
                continue
            rows.append(
                {
                    "stock": stock,
                    "earnings_date": reported,
                    "fiscal_date_ending": fiscal,
                }
            )

        df = pd.DataFrame(rows, columns=["stock", "earnings_date", "fiscal_date_ending"])
        return df.sort_values(["stock", "earnings_date"]).reset_index(drop=True)

    # should never reach here
    raise RuntimeError(f"Alpha Vantage failed for {stock}: {last_err}")

def fetch_earnings_dates(sleep_between = 0.0, deduplicate = True) -> pd.DataFrame:
    """
        Fetch earnings dates for multiple stocks and stack into one DataFrame.
    """
    stocks_list = read_stocks_to_fetch()
    print(f"{len(stocks_list)} Stocks to fetch.\n")
    
    if Path(EARNINGS_PATH).exists():
        print(f"Using cached Earnings Data from {EARNINGS_PATH}\n")
        return pd.read_csv(EARNINGS_PATH)
        
    print("No cached Earnings data, fetching NEW...")
    print("Fetching Earnings Dates")
    frames: List[pd.DataFrame] = []
    for stock in stocks_list:
        print(f"Fetching {stock} Earnings")
        df = fetch_earnings_dates_for_stock(stock)
        if not df.empty:
            frames.append(df)
        if sleep_between > 0:
            time.sleep(sleep_between)

    if not frames:
        return pd.DataFrame(columns=["stock", "earnings_date", "fiscal_date_ending"])

    earnings_df = pd.concat(frames, ignore_index=True)

    if deduplicate:
        earnings_df = (
            earnings_df.sort_values(["stock", "earnings_date"])
            .drop_duplicates(subset=["stock", "earnings_date"], keep="first")
            .reset_index(drop=True)
        )

    earnings_df.to_csv("data/earnings_dates.csv", index=False)
    print(f"Saved Earnings: {EARNINGS_PATH}")
    print(f"Rows: {len(earnings_df):,}")
    print(f"Stocks with data: {earnings_df['stock'].nunique():,}\n")
    return earnings_df
