import requests
import os 
import pandas as pd
from pathlib import Path

from config import (ALPHAVANTAGE_BASE_URL,
                    STOCKS_START_DATE,
                    EPS_PATH,
                    USE_CACHED_DATA_FLAG,
                    STOCK_LIST_PATH)

from data_utilities.helper_funcs import get_alpha_vantage_api_key, read_stocks_to_fetch

def fetch_eps_single_stock(stock: str ) -> dict:
    """
        Fetch EPS data for a list of stocks from Alpha Vantage.
    """
    api_key = get_alpha_vantage_api_key()

    url = (f"{ALPHAVANTAGE_BASE_URL}"
        f"?function=EARNINGS"
        f"&symbol={stock}"
        f"&apikey={api_key}"
        )
    try:
        r = requests.get(url, timeout = 30)
        r.raise_for_status() # Raise an exception for bad HTTP status codes
        data = r.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"[EPS] HTTP error for {stock}") from e
    except ValueError as e:
        raise RuntimeError(f"[EPS] Invalid JSON for {stock}") from e

    if "Note" in data or "premium" in data:
        raise RuntimeError(f"[EPS] Rate limited by Alpha Vantage: {stock}")
    if "Information" in data:
        raise RuntimeError(f"[EPS] API info for {stock}: {data['Information']}")
    if "Error" in data:
        raise RuntimeError(f"[EPS] API error for {stock}")
    if "quarterlyEarnings" not in data:
        raise RuntimeError(f"[EPS] Missing earnings data for {stock}")
    
    return data

def parse_quarterly_eps(data: dict) -> pd.DataFrame:
    quarterly = data.get("quarterlyEarnings", [])
    if not quarterly:
        return pd.DataFrame(
            columns=["stock", "fiscal_date", "reported_date", "reported_eps", "estimated_eps", "surprisePercentage"]
        )

    df = pd.DataFrame(quarterly)
    df = df.rename(columns={
        "fiscalDateEnding": "fiscal_date",
        "reportedDate": "reported_date",
        "reportedEPS": "reported_eps",
        "estimatedEPS": "estimated_eps",
        "surprisePercentage": "surprise_percentage"
    })

    df["stock"] = data.get("symbol") 
    df["fiscal_date"] = pd.to_datetime(df["fiscal_date"])
    df["reported_date"] = pd.to_datetime(df["reported_date"], errors="coerce")

    df["reported_eps"] = pd.to_numeric(df["reported_eps"], errors="coerce")
    df["estimated_eps"] = pd.to_numeric(df["estimated_eps"], errors="coerce")
    df["surprise_percentage"] = pd.to_numeric(df["surprise_percentage"], errors="coerce")

    df["surprise_percentage"] = df["surprise_percentage"] / 100.0
    df = df[df["fiscal_date"] >= STOCKS_START_DATE]
    df = df[["stock", "fiscal_date", "reported_date", "reported_eps", "estimated_eps", "surprise_percentage"]]

    return df

def fetch_eps() -> pd.DataFrame:
    stocks = read_stocks_to_fetch()
    if USE_CACHED_DATA_FLAG == True:
        if Path(EPS_PATH).exists():
            print(f"Using cached EPS Data from {EPS_PATH}\n")
            return pd.read_csv(EPS_PATH)
        
        
    print("No cached EPS data, fetching NEW...")
    all_eps_data = []
    errors = []
    for i,stock in enumerate(stocks, start=1):
        print(f"Fetching {stock} EPS ({i}/{len(stocks)})")
        try:
            data = fetch_eps_single_stock(stock)
            df = parse_quarterly_eps(data)
            all_eps_data.append(df)
        except RuntimeError as e:
            errors.append({
                "stock": stock,
                "error": str(e)
            })
            continue

    if all_eps_data:
        eps_df = pd.concat(all_eps_data, ignore_index=True)
    else:
        eps_df = pd.DataFrame(
            columns=["stock", "fiscal_date", "reported_date", "reported_eps", "estimated_eps", "surprise_percentage"]
        )

    eps_df.to_csv(EPS_PATH, index=False)
    print("Errors:", errors)
    return eps_df
