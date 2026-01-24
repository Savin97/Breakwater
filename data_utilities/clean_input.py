""" Cleans Inputs """
import pandas as pd
from pathlib import Path

def read_stocks_to_fetch(path: Path) -> list[str]:
    """
        Reads stocks from a file. Supports:
        - .txt: one stock per line
        - .csv: column named symbol/ticker/stock 
        Returns a list of all unique stocks (uppercase, no spaces)
    """
    if not path.exists():
        raise FileNotFoundError(f"Stocks file not found: {path}")

    if path.suffix.lower() == ".csv":
        stock_prices_df = pd.read_csv(path)
        col = None
        for c in ("stock", "symbol", "ticker","Stock", "Symbol", "Ticker"):
            if c in stock_prices_df.columns:
                col = c
                break
        if col is None:
            raise ValueError(f"CSV must contain a symbol/ticker/stock column. Found: {list(stock_prices_df.columns)}")
        stocks = stock_prices_df[col].astype(str).str.strip().tolist()
    else:
        print("Reading stocks from text file")
        stocks = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        print(stocks)

    # Basic cleanup
    stocks = [t.replace(" ", "").upper() for t in stocks if t]
    # dedupe preserve order
    seen = set()
    out = []
    for stock in stocks:
        if stock and stock != "NAN" and stock not in seen:
            out.append(stock)
            seen.add(stock)
    return out