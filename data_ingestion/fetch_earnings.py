"""
fetch_earnings.py

Fetch historical quarterly earnings *report dates* from Alpha Vantage.

- Provider: Alpha Vantage
- Output columns:
    stock, earnings_date, fiscal_date_ending

Usage:
    from data_ingestion.fetch_earnings import fetch_earnings_dates

    df = fetch_earnings_dates(["AAPL", "MSFT"], start_date=STOCKS_START_DATE)
    print(df.head())

Auth:
    Set env var: ALPHAVANTAGE_API_KEY="..."
"""
import time
import requests
import pandas as pd
from typing import Iterable, Optional, List
from pathlib import Path

from config import (ALPHAVANTAGE_BASE_URL, 
                    FREE_ALPHAVANTAGE_KEY,
                    STOCKS_START_DATE,
                    TIMEOUT_SECONDS,
                    MAX_RETRIES,
                    BACKOFF_SECONDS,
                    EARNINGS_PATH,
                    USE_CACHED_DATA_FLAG)
from data_utilities.formatting import parse_date


def fetch_earnings_dates_for_stock(
    stock: str,
    start_date: str = STOCKS_START_DATE,
    api_key: Optional[str] = None
        ) -> pd.DataFrame:
    """
        Fetch quarterly earnings report dates for a single stock from Alpha Vantage.
        Keeps only rows where reportedDate >= start_date.
    """
    # TODO: Use env variable
    #api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    api_key = FREE_ALPHAVANTAGE_KEY
    if not api_key:
        raise ValueError(
            "Missing Alpha Vantage API key. Set ALPHAVANTAGE_API_KEY or pass api_key=..."
        )

    start_dt = parse_date(start_date)
    params = {"function": "EARNINGS", "symbol": stock, "apikey": api_key}

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(ALPHAVANTAGE_BASE_URL, params=params, timeout=TIMEOUT_SECONDS)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            last_err = exc
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Alpha Vantage request failed for {stock}: {exc}")
            print(f"Waiting {BACKOFF_SECONDS * attempt} seconds")
            time.sleep(BACKOFF_SECONDS * attempt)
            continue

        # Rate-limit / errors
        if "Note" in data:
            last_err = RuntimeError(data["Note"])
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Alpha Vantage rate limit for {stock}: {data['Note']}")
            print(f"Waiting {BACKOFF_SECONDS * attempt} seconds")
            time.sleep(BACKOFF_SECONDS * attempt)
            continue

        if "Error Message" in data:
            raise RuntimeError(f"Alpha Vantage error for {stock}: {data['Error Message']}")

        quarterly = data.get("quarterlyEarnings", []) or []
        rows = []
        for q in quarterly:
            reported = parse_date(q.get("reportedDate"))
            fiscal = parse_date(q.get("fiscalDateEnding"))
            if reported is None:
                continue
            if reported < start_dt:
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

def fetch_earnings_dates(
        stocks: Iterable[str],
        start_date: str = STOCKS_START_DATE,
        api_key: Optional[str] = None,
        sleep_between: float = 0.0,
        deduplicate: bool = True,
    ) -> pd.DataFrame:
    """
        Fetch earnings dates for multiple stocks and stack into one DataFrame.
    """
    if USE_CACHED_DATA_FLAG == True:
        if Path(EARNINGS_PATH).exists():
            print(f"\nUsing cached Earnings Data from {EARNINGS_PATH}\n")
            return pd.read_csv(EARNINGS_PATH)
        
    print("No cached Earnings data, fetching NEW...")
    print("Fetching Earnings Dates")
    frames: List[pd.DataFrame] = []
    for stock in stocks:
        print(f"Fetching {stock} Earnings")
        df = fetch_earnings_dates_for_stock(stock, start_date=start_date, api_key=api_key)
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
