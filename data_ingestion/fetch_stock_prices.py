#fetch_stock_prices.py
""" 
    Fetch stock prices going back to 2008
    Outputs a simple table:
    stock | date | price (adj closed price)
"""
from config import ( STOCKS_START_DATE,
                    STOCKS_END_DATE, 
                    PRICES_PATH, 
                    STOCK_NAMES_FILE_PATH,
                    TIMEOUT_SECONDS,
                    BACKOFF_SECONDS,
                    MAX_RETRIES,
                    DEFAULT_FETCH_CHUNK_SIZE,
                    USE_CACHED_DATA_FLAG )
import sys
import time
from pathlib import Path
import pandas as pd
import yfinance as yf


from data_utilities.clean_input import read_stocks_to_fetch
from data_utilities.helper_funcs import chunk_list, ensure_parent_dir, sleep_backoff

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

    
def fetch_stock_prices(provider: str) -> pd.DataFrame:
    """
        Fetch stock prices for a list of stocks from a specified provider.
        Outputs a DF: stock | date | price
    """
    stocks = read_stocks_to_fetch(Path(STOCK_NAMES_FILE_PATH))

    if not stocks:
        raise ValueError("No stocks found.")

    # TODO: Temp solution, caching should work differently in the production version
    if USE_CACHED_DATA_FLAG == True:
        if Path(PRICES_PATH).exists():
            print(f"\nUsing cached Prices from {PRICES_PATH}\n")
            return pd.read_csv(PRICES_PATH)
            
    print("No cached Prices data, fetching NEW...")        
    print(f"Provider: {provider}")
    print(f"Stocks: {len(stocks)}")
    print(f"Date range: {STOCKS_START_DATE} → {STOCKS_END_DATE}")
    print(f"Chunk size: {DEFAULT_FETCH_CHUNK_SIZE}")

    parts = []
    done = 0
    total = len(stocks)

    for i,batch in enumerate(chunk_list(stocks, DEFAULT_FETCH_CHUNK_SIZE), start=1):
        if provider == "yfinance":
            print(f"Fetching Stock Prices in chunks, batch number {i}")
            df_batch = fetch_stocks_yfinance(
                stocks=batch
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        parts.append(df_batch)
        done += len(batch)
        got = df_batch["stock"].nunique() if not df_batch.empty else 0
        print(f"Fetched prices for {got}/{len(batch)} stocks in this batch. Progress attempted {done}/{total}")


        time.sleep(TIMEOUT_SECONDS)

    prices_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["stock", "date", "price"]
    )

    if prices_df.empty:
        raise RuntimeError("No data fetched (blocked/rate-limited or bad stocks list).")

    prices_df = prices_df.drop_duplicates(subset=["stock", "date"], keep="last")

    out_path = Path(PRICES_PATH)
    ensure_parent_dir(out_path)
    prices_df.to_csv(out_path, index=False)

    print(f"Saved Prices: {out_path}")
    print(f"Rows: {len(prices_df):,}")
    print(f"Stocks with data: {prices_df['stock'].nunique():,}\n")
    return prices_df
