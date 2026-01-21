#fetch_stock_prices.py
""" 
    Fetch stock prices going back to 2008
    Outputs a simple table:
    symbol | date | price_adj_closed_price
"""
from config import ( TICKERS_START_DATE,
                    TICKERS_END_DATE, 
                    PRICES_PATH, 
                    TICKERS_FILE_PATH,
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


from data_utilities.clean_input import read_tickers_to_fetch
from data_utilities.helper_funcs import chunk_list, ensure_parent_dir, sleep_backoff

# Provider implementations
def fetch_tickers_yfinance(tickers: list[str]) -> pd.DataFrame:
    """
        Uses yfinance to fetch stock price data
        Returns a tidy DF: symbol | date | price_adj_close
        Notes:
        - yfinance returns "Adj Close" adjusted for splits/dividends (Yahoo's adjusted close).
        - We only keep Adj Close. If it's missing, we fall back to Close (with a warning).
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed. Run pip install yfinance")

    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            raw = yf.download(
                tickers=tickers,
                start=TICKERS_START_DATE,
                end=TICKERS_END_DATE,
                interval="1d",
                group_by="ticker",
                auto_adjust=False,   # keep Adj Close column
                actions=False,
                threads=True,
                progress=False,
            )

            if raw is None or raw.empty:
                raise RuntimeError("Empty response from yfinance (rate-limit/block/bad tickers).")

            rows = []

            # Multi ticker: columns MultiIndex (ticker, field)
            if isinstance(raw.columns, pd.MultiIndex):
                available_syms = set(raw.columns.get_level_values(0))
                for sym in tickers:
                    if sym not in available_syms:
                        continue
                    sub = raw[sym].copy()
                    price_col = "Adj Close" if "Adj Close" in sub.columns else "Close"
                    if price_col not in sub.columns:
                        continue

                    s = sub[price_col].rename("price_adj_close").dropna()
                    if s.empty:
                        continue

                    df_sym = s.reset_index().rename(columns={"Date": "date"})
                    df_sym["symbol"] = sym
                    rows.append(df_sym[["symbol", "date", "price_adj_close"]])

            # Single ticker: columns are fields
            else:
                price_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
                s = raw[price_col].rename("price_adj_close").dropna()
                df_sym = s.reset_index().rename(columns={"Date": "date"})
                df_sym["symbol"] = tickers[0]
                rows.append(df_sym[["symbol", "date", "price_adj_close"]])

            out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
                columns=["symbol", "date", "price_adj_close"]
            )

            out["date"] = pd.to_datetime(out["date"]).dt.date
            out["price_adj_close"] = pd.to_numeric(out["price_adj_close"], errors="coerce")
            out = out.dropna(subset=["price_adj_close"])
            out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
            return out

        except Exception as e:
            last_err = e
            print(f"[yfinance] attempt {attempt+1}/{MAX_RETRIES + 1} failed: {e}", file=sys.stderr)
            if attempt < MAX_RETRIES:
                sleep_backoff(attempt, BACKOFF_SECONDS)

    raise RuntimeError(f"yfinance failed after retries: {last_err}")

    
def fetch_stock_prices(provider: str) -> None:
    """
        Fetch stock prices for a list of tickers from a specified provider.
        Outputs a DF: symbol | date | price_adj_close
    """
    tickers = read_tickers_to_fetch(Path(TICKERS_FILE_PATH))

    if not tickers:
        raise ValueError("No tickers found.")
    
    # TODO: Temp solution, caching should work differently in the production version
    if Path(PRICES_PATH).exists() and USE_CACHED_DATA_FLAG:
        print(f"\nUsing cached Prices from {PRICES_PATH}\n")
        return pd.read_csv(PRICES_PATH)

    print(f"Provider: {provider}")
    print(f"Tickers: {len(tickers)}")
    print(f"Date range: {TICKERS_START_DATE} → {TICKERS_END_DATE}")
    print(f"Chunk size: {DEFAULT_FETCH_CHUNK_SIZE}")

    parts = []
    done = 0
    total = len(tickers)

    for batch in chunk_list(tickers, DEFAULT_FETCH_CHUNK_SIZE):
        if provider == "yfinance":
            df_batch = fetch_tickers_yfinance(
                tickers=batch
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        parts.append(df_batch)
        done += len(batch)
        print(f"Fetched prices for {len(batch)} tickers. Progress {done}/{total}")

        time.sleep(TIMEOUT_SECONDS)

    prices_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["symbol", "date", "price_adj_close"]
    )

    if prices_df.empty:
        raise RuntimeError("No data fetched (blocked/rate-limited or bad tickers list).")

    prices_df = prices_df.drop_duplicates(subset=["symbol", "date"], keep="last")

    out_path = Path(PRICES_PATH)
    ensure_parent_dir(out_path)
    prices_df.to_csv(out_path, index=False)

    print(f"Saved Prices: {out_path}")
    print(f"Rows: {len(prices_df):,}")
    print(f"Tickers with data: {prices_df['symbol'].nunique():,}\n")
    return prices_df
