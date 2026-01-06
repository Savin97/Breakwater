# fetch_stock_prices.py

import sys, time
import pandas as pd
from datetime import date

try:
    import yfinance as yf
except Exception:
    yf = None

TICKERS_START_DATE, TICKERS_END_DATE = "2008-01-01", date.today().isoformat()

def chunk_list(items: list[str], n: int):
    for i in range(0, len(items), n):
        yield items[i:i+n]


def sleep_backoff(attempt: int, base: float) -> None:
    wait = base * (2 ** attempt)
    wait = wait + (0.1 * wait)
    time.sleep(wait)


def _fetch_yfinance_batch(
    tickers: list[str],
    start: str,
    end: str,
    max_retries: int = 5,
    base_backoff_sec: float = 2.0,
) -> pd.DataFrame:
    """Internal: fetch one batch of tickers from yfinance."""
    if yf is None:
        raise RuntimeError("yfinance is not installed. Run pip install yfinance")

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            raw = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                actions=False,
                threads=True,
                progress=False,
            )

            if raw is None or raw.empty:
                raise RuntimeError("Empty response from yfinance.")

            rows = []

            if isinstance(raw.columns, pd.MultiIndex):
                available_syms = set(raw.columns.get_level_values(0))
                for sym in tickers:
                    if sym not in available_syms:
                        continue
                    sub = raw[sym].copy()
                    price_col = "Adj Close" if "Adj Close" in sub.columns else "Close"
                    if price_col not in sub.columns:
                        continue

                    s = sub[price_col].rename("adj_close").dropna()
                    if s.empty:
                        continue

                    df_sym = s.reset_index().rename(columns={"Date": "date"})
                    df_sym["symbol"] = sym
                    rows.append(df_sym[["symbol", "date", "adj_close"]])

            else:
                price_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
                s = raw[price_col].rename("adj_close").dropna()
                df_sym = s.reset_index().rename(columns={"Date": "date"})
                df_sym["symbol"] = tickers[0]
                rows.append(df_sym[["symbol", "date", "adj_close"]])

            out = pd.concat(rows, ignore_index=True)
            out["date"] = pd.to_datetime(out["date"]).dt.date
            out["adj_close"] = pd.to_numeric(out["adj_close"], errors="coerce")
            out = out.dropna(subset=["adj_close"])
            out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
            out.to_csv("debug_yf_output.csv", index=False)  # Debug line
            return out

        except Exception as e:
            last_err = e
            print(f"[yfinance] attempt {attempt+1}/{max_retries+1} failed: {e}", file=sys.stderr)
            if attempt < max_retries:
                sleep_backoff(attempt, base_backoff_sec)

    raise RuntimeError(f"yfinance failed after retries: {last_err}")


def fetch_prices_for_tickers(
    tickers: list[str],
    chunk_size: int = 50,
    start: str = TICKERS_START_DATE,
    end: str = TICKERS_END_DATE,
) -> pd.DataFrame:
    """Public function you use in the pipeline."""
    parts = []
    for batch in chunk_list(tickers, chunk_size):
        df_batch = _fetch_yfinance_batch(batch, start=start, end=end)
        parts.append(df_batch)

    if not parts:
        return pd.DataFrame(columns=["symbol", "date", "adj_close"])

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["symbol", "date"], keep="last")
    return df
tickers = ["AAPL", "MSFT", "GOOGL"]
df = fetch_prices_for_tickers(tickers)
df.to_csv("output.csv", index=False)