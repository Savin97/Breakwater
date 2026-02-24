#fetch_stock_prices.py
""" 
    Fetch stock prices going back to 2008
    Outputs a simple table:
    stock | date | price (adj closed price)
"""
from config import ( STOCKS_START_DATE,
                    STOCKS_END_DATE, 
                    PRICES_PATH, 
                    STOCK_LIST_PATH,
                    TIMEOUT_SECONDS,
                    BACKOFF_SECONDS,
                    MAX_RETRIES,
                    DEFAULT_FETCH_CHUNK_SIZE,
                    ALPHAVANTAGE_CALLS_PER_MINUTE)
import sys
import time
import requests
from pathlib import Path
import pandas as pd
import yfinance as yf

from data_utilities.formatting import parse_date
from data_utilities.helper_funcs import (chunk_list, 
                                        sleep_backoff,
                                        get_alpha_vantage_api_key,
                                        check_cached_data_use,
                                        read_stocks_to_fetch)
def fetch_stock_prices(provider: str) -> pd.DataFrame:
    """
        Fetch stock prices for a list of stocks from a specified provider.
        Outputs a DF: stock | date | price
    """
    stock_list = read_stocks_to_fetch()

    if not stock_list:
        raise ValueError("No stocks found.")

    # TODO: Temp solution, caching should work differently in the production version
    if Path(PRICES_PATH).exists():
        print(f"Using cached Prices from {PRICES_PATH}\n")
        return pd.read_csv(PRICES_PATH)
            
    print("No cached Prices data, fetching NEW...")        
    print(f"Provider: {provider}")
    print(f"Stocks: {len(stock_list)}")
    print(f"Date range: {STOCKS_START_DATE} -> {STOCKS_END_DATE}")

    parts = []
    done = 0
    total = len(stock_list)

    if provider == "ALPHAVANTAGE":
            print(f"Fetching Stock Prices...")
            api_key = get_alpha_vantage_api_key()
            prices_df = fetch_stocks_alpha_vantage(stock_list, api_key, outputsize="full")

    elif provider == "yfinance":
        for i,batch in enumerate(chunk_list(stock_list, DEFAULT_FETCH_CHUNK_SIZE), start=1):
            print(f"Fetching Stock Prices in chunks, batch number {i}")
            print(f"Chunk size: {DEFAULT_FETCH_CHUNK_SIZE}")
            df_batch = fetch_stocks_yfinance(
                stocks=batch
            )
            parts.append(df_batch)
            done += len(batch)
            got = df_batch["stock"].nunique() if not df_batch.empty else 0
            print(f"Fetched prices for {got}/{len(batch)} stocks in this batch. Progress attempted {done}/{total}")

            time.sleep(TIMEOUT_SECONDS)

        prices_df = (
            pd.concat(parts, ignore_index=True) if parts 
            else pd.DataFrame(columns=["stock", "date", "price"])
        )

            
    else:
        prices_df = pd.DataFrame()
        raise ValueError(f"Unknown provider: {provider}")
        
    if prices_df.empty:
        raise RuntimeError("No data fetched (blocked/rate-limited or bad stocks list).")

    prices_df = prices_df.drop_duplicates(subset=["stock", "date"], keep="last")

    out_path = Path(PRICES_PATH)
    prices_df.to_csv(out_path, index=False)

    print(f"Saved Prices: {out_path}")
    print(f"Rows: {len(prices_df):,}")
    print(f"Stocks with data: {prices_df['stock'].nunique():,}\n")
    return prices_df


BASE_URL = "https://www.alphavantage.co/query"

def fetch_stocks_alpha_vantage(stocks, api_key, outputsize="compact"):
    """
        Fetch daily adjusted close prices from Alpha Vantage.
        Returns DF: stock | date | price
    """
    start_date = parse_date(STOCKS_START_DATE)
    end_date = parse_date(STOCKS_END_DATE)
    min_sleep = 60 / ALPHAVANTAGE_CALLS_PER_MINUTE  # ~0.8s for 75/min
    parts = []

    for i, stock in enumerate(stocks, start=1):
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": stock,
            "outputsize": outputsize,   # "compact" (~100 days) or "full"
            "apikey": api_key
        }

        r = requests.get(BASE_URL, params=params, timeout=TIMEOUT_SECONDS)
        data = r.json()

        # Basic failure handling (AlphaVantage often returns errors inside JSON)
        if "Error Message" in data or "Time Series (Daily)" not in data:
            print(f"[{i}] {stock}: no data / error")
            time.sleep(min_sleep)
            continue

        if ("Note" in data) or ("Information" in data):
            msg = data.get("Note") or data.get("Information")
            print(f"[{i}] {stock}: throttled by AlphaVantage: {msg}")
            time.sleep(BACKOFF_SECONDS) 
            continue

        ts = data["Time Series (Daily)"]

        rows = []
        for date_str, ohlc in ts.items():
            price = float(ohlc["5. adjusted close"])
            rows.append((stock, date_str, price))

        prices_df = pd.DataFrame(rows, columns=["stock", "date", "price"])
        prices_df["date"] = parse_date(prices_df["date"])

        # 🔑 DATE FILTERING (this is what you wanted)
        if start_date is not None:
            prices_df = prices_df[prices_df["date"] >= start_date]
        if end_date is not None:
            prices_df = prices_df[prices_df["date"] <= end_date]

        parts.append(prices_df)
        # TODO: change to i/len(stocks)
        print(f"{i}/{len(stocks)} {stock}: {len(prices_df)} rows")
        time.sleep(min_sleep)

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["stock", "date", "price"])
    out = out.drop_duplicates(["stock", "date"]).sort_values(["stock", "date"]).reset_index(drop=True)
    return out





# Provider implementations
def fetch_stocks_yfinance(stocks: list[str]) -> pd.DataFrame:
    """
        Uses yfinance to fetch stock price (adj_close) data
        Returns a tidy DF: stock | date | price
        Notes:
        - yfinance returns "Adj Close" adjusted for splits/dividends (Yahoo's adjusted close).
        - We only keep Adj Close. If it's missing, we fall back to Close (with a warning).
    """
    def _download(tickers: list[str]):
        return yf.download(
            tickers=tickers,
            start=STOCKS_START_DATE,
            end=STOCKS_END_DATE,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            actions=False,
            threads=False,   
            progress=False,
        )
    
    if yf is None:
        raise RuntimeError("yfinance is not installed. Run pip install yfinance")

    last_err = None
    for attempt in range(MAX_RETRIES + 1):

        try:
            print(f"Fetching Stock Prices in Bulk...")
            raw = _download(stocks)

            if raw is None or raw.empty:
                raise RuntimeError("Empty response from yfinance (rate-limit/block/bad stocks).")

            rows = []

            # Multi stock: columns MultiIndex (stock, field)
            if isinstance(raw.columns, pd.MultiIndex):
                available_stocks = set(raw.columns.get_level_values(0))
                missing = [s for s in stocks if s not in available_stocks]
                if missing:
                    raise RuntimeError(f"yfinance returned partial data; missing: {missing}")
                
                # Parse what we get
                for stock in stocks:
                    if stock not in available_stocks:
                        continue
                    sub = raw[stock].copy()
                    price_col = "Adj Close" if "Adj Close" in sub.columns else "Close"
                    if price_col not in sub.columns:
                        continue

                    s = sub[price_col].rename("price").dropna()
                    if s.empty:
                        continue

                    df_sym = s.reset_index().rename(columns={"Date": "date"})
                    df_sym["stock"] = stock
                    rows.append(df_sym[["stock", "date", "price"]])

                # retry missing one-by-one (best reliability)
                for m in missing:
                    print(f"Retrying missing ticker individually: {m}")
                    raw_m = _download([m])
                    if raw_m is None or raw_m.empty:
                        continue
                    price_col = "Adj Close" if "Adj Close" in raw_m.columns else "Close"
                    if price_col not in raw_m.columns:
                        continue
                    s = raw_m[price_col].rename("price").dropna()
                    if s.empty:
                        continue
                    df_m = s.reset_index().rename(columns={"Date": "date"})
                    df_m["stock"] = m
                    rows.append(df_m[["stock", "date", "price"]])
            
            # Single stock: columns are fields
            else:
                price_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
                s = raw[price_col].rename("price").dropna()
                df_sym = s.reset_index().rename(columns={"Date": "date"})
                df_sym["stock"] = stocks[0]
                rows.append(df_sym[["stock", "date", "price"]])
            
            out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
                columns=["stock", "date", "price"]
            )

            out["date"] = pd.to_datetime(out["date"]).dt.date
            out["price"] = pd.to_numeric(out["price"], errors="coerce")
            out = out.dropna(subset=["price"])
            out = out.sort_values(["stock", "date"]).reset_index(drop=True)
            return out

        except Exception as e:
            last_err = e
            print(f"[yfinance] attempt {attempt+1}/{MAX_RETRIES + 1} failed: {e}", file=sys.stderr)
            if attempt < MAX_RETRIES:
                sleep_backoff(attempt)

    raise RuntimeError(f"yfinance failed after retries: {last_err}")


