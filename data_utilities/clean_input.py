""" Cleans Inputs """
import pandas as pd
from pathlib import Path

def read_tickers_to_fetch(path: Path) -> list[str]:
    """
        Reads tickers from a file. Supports:
        - .txt: one ticker per line
        - .csv: column named symbol/ticker/stock 
        Returns a list of all unique tickers (uppercase, no spaces)
    """
    if not path.exists():
        raise FileNotFoundError(f"Tickers file not found: {path}")

    if path.suffix.lower() == ".csv":
        stock_prices_df = pd.read_csv(path)
        col = None
        for c in ("symbol", "ticker", "Symbol", "Ticker"):
            if c in stock_prices_df.columns:
                col = c
                break
        if col is None:
            raise ValueError(f"CSV must contain a symbol/ticker/stock column. Found: {list(stock_prices_df.columns)}")
        tickers = stock_prices_df[col].astype(str).str.strip().tolist()
    else:
        print("Reading tickers from text file")
        tickers = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        print(tickers)

    # Basic cleanup
    tickers = [t.replace(" ", "").upper() for t in tickers if t]
    # dedupe preserve order
    seen = set()
    out = []
    for t in tickers:
        if t and t != "NAN" and t not in seen:
            out.append(t)
            seen.add(t)
    print(f"Tickers to fetch: {out}")
    return out