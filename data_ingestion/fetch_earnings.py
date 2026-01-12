"""
fetch_earnings.py

Fetch historical quarterly earnings *report dates* from Alpha Vantage.

- Provider: Alpha Vantage (function=EARNINGS)
- Output columns:
    stock, earnings_date, fiscal_date_ending

Usage:
    from data_ingestion.fetch_earnings import fetch_earnings_dates

    df = fetch_earnings_dates(["AAPL", "MSFT"], start_date="2008-01-01")
    print(df.head())

Auth:
    Set env var: ALPHAVANTAGE_API_KEY="..."
"""

from __future__ import annotations

import os
import time
import requests
import pandas as pd
from typing import Iterable, Optional, List

from config import (ALPHAVANTAGE_BASE_URL,
                    TICKERS_START_DATE,
                    TIMEOUT_SECONDS,
                    MAX_RETRIES,
                    BACKOFF_SECONDS)
from data_utilities.formatting import parse_date


def fetch_earnings_dates_for_symbol(
    symbol: str,
    start_date: str = TICKERS_START_DATE,
    api_key: Optional[str] = None,
    timeout_seconds: float = TIMEOUT_SECONDS,
    retries: int = MAX_RETRIES,
    backoff_seconds: float = BACKOFF_SECONDS,
    ) -> pd.DataFrame:
    """
        Fetch quarterly earnings report dates for a single symbol from Alpha Vantage.

        Keeps only rows where reportedDate >= start_date.
    """

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing Alpha Vantage API key. Set ALPHAVANTAGE_API_KEY or pass api_key=..."
        )

    start_dt = parse_date(start_date)

    params = {"function": "EARNINGS", "symbol": symbol, "apikey": api_key}

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(ALPHAVANTAGE_BASE_URL, params=params, timeout=timeout_seconds)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            last_err = exc
            if attempt == retries:
                raise RuntimeError(f"Alpha Vantage request failed for {symbol}: {exc}")
            time.sleep(backoff_seconds * attempt)
            continue

        # Rate-limit / errors
        if "Note" in data:
            last_err = RuntimeError(data["Note"])
            if attempt == retries:
                raise RuntimeError(f"Alpha Vantage rate limit for {symbol}: {data['Note']}")
            time.sleep(backoff_seconds * attempt)
            continue

        if "Error Message" in data:
            raise RuntimeError(f"Alpha Vantage error for {symbol}: {data['Error Message']}")

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
                    "stock": symbol,
                    "earnings_date": reported,
                    "fiscal_date_ending": fiscal,
                }
            )

        df = pd.DataFrame(rows, columns=["stock", "earnings_date", "fiscal_date_ending"])
        return df.sort_values(["stock", "earnings_date"]).reset_index(drop=True)

    # should never reach here
    raise RuntimeError(f"Alpha Vantage failed for {symbol}: {last_err}")

def fetch_earnings_dates(
        symbols: Iterable[str],
        start_date: str = TICKERS_START_DATE,
        api_key: Optional[str] = None,
        sleep_between: float = 0.0,
        deduplicate: bool = True,
    ) -> pd.DataFrame:
    """
        Fetch earnings dates for multiple symbols and stack into one DataFrame.
    """


    
    frames: List[pd.DataFrame] = []
    for sym in symbols:
        df = fetch_earnings_dates_for_symbol(sym, start_date=start_date, api_key=api_key)
        if not df.empty:
            frames.append(df)
        if sleep_between > 0:
            time.sleep(sleep_between)

    if not frames:
        return pd.DataFrame(columns=["stock", "earnings_date", "fiscal_date_ending"])

    out = pd.concat(frames, ignore_index=True)

    if deduplicate:
        out = (
            out.sort_values(["stock", "earnings_date"])
            .drop_duplicates(subset=["stock", "earnings_date"], keep="first")
            .reset_index(drop=True)
        )
    out.to_csv("data/earnings_dates.csv", index=False)
    return out


if __name__ == "__main__":
    # Quick manual test:
    #   python fetch_earnings.py AAPL MSFT
    import sys

    tickers = sys.argv[1:] or ["AAPL"]
    df = fetch_earnings_dates(tickers, start_date="2008-01-01", sleep_between=1.0)
    print(df.head(20))
    print("Rows:", len(df))