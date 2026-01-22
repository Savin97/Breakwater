import requests
import os 
import pandas as pd
from pathlib import Path

from config import ( ALPHAVANTAGE_BASE_URL, 
                    FREE_ALPHAVANTAGE_KEY, 
                    TICKERS_START_DATE,
                    EPS_PATH,
                    USE_CACHED_DATA_FLAG )
from data_utilities.clean_input import read_tickers_to_fetch

def get_alpha_vantage_api_key() -> str:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing ALPHAVANTAGE_API_KEY environment variable"
        )
    return api_key

def fetch_eps_single_ticker(ticker: str ) -> dict:
    """
        Fetch EPS data for a list of tickers from Alpha Vantage.
    """
    # api_key = get_alpha_vantage_api_key()

    api_key = FREE_ALPHAVANTAGE_KEY
    url = (f"{ALPHAVANTAGE_BASE_URL}"
        f"?function=EARNINGS"
        f"&symbol={ticker}"
        f"&apikey={api_key}"
        )
    try:
        r = requests.get(url, timeout = 30)
        r.raise_for_status() # Raise an exception for bad HTTP status codes
        data = r.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"[EPS] HTTP error for {ticker}") from e
    except ValueError as e:
        raise RuntimeError(f"[EPS] Invalid JSON for {ticker}") from e

    if "Note" in data or "premium" in data:
        raise RuntimeError(f"[EPS] Rate limited by Alpha Vantage: {ticker}")
    if "Information" in data:
        raise RuntimeError(f"[EPS] API info for {ticker}: {data['Information']}")
    if "Error" in data:
        raise RuntimeError(f"[EPS] API error for {ticker}")
    if "quarterlyEarnings" not in data:
        raise RuntimeError(f"[EPS] Missing earnings data for {ticker}")
    
    return data

def parse_quarterly_eps(data: dict) -> pd.DataFrame:
    quarterly = data.get("quarterlyEarnings", [])
    if not quarterly:
        return pd.DataFrame(
            columns=["symbol", "fiscal_date", "reported_date", "reported_eps", "estimated_eps", "surprisePercentage"]
        )

    df = pd.DataFrame(quarterly)
    df = df.rename(columns={
        "fiscalDateEnding": "fiscal_date",
        "reportedDate": "reported_date",
        "reportedEPS": "reported_eps",
        "estimatedEPS": "estimated_eps",
        "surprisePercentage": "surprise_percentage"
    })

    df["symbol"] = data.get("symbol")
    df["fiscal_date"] = pd.to_datetime(df["fiscal_date"])
    df["reported_date"] = pd.to_datetime(df["reported_date"], errors="coerce")

    df["reported_eps"] = pd.to_numeric(df["reported_eps"], errors="coerce")
    df["estimated_eps"] = pd.to_numeric(df["estimated_eps"], errors="coerce")
    df["surprise_percentage"] = pd.to_numeric(df["surprise_percentage"], errors="coerce")

    df["surprise_percentage"] = df["surprise_percentage"] / 100.0
    df = df[df["fiscal_date"] >= TICKERS_START_DATE]
    df = df[["symbol", "fiscal_date", "reported_date", "reported_eps", "estimated_eps", "surprise_percentage"]]

    return df

def fetch_eps(tickers: list) -> pd.DataFrame:

    if USE_CACHED_DATA_FLAG == True:
        if Path(EPS_PATH).exists():
            print(f"\nUsing cached EPS Data from {EPS_PATH}\n")
            return pd.read_csv(EPS_PATH)
        
        
    print("No cached EPS data, fetching NEW...")
    all_eps_data = []
    errors = []
    for i,ticker in enumerate(tickers, start=1):
        print(f"Fetching {ticker} EPS ({i}/{len(tickers)})")
        try:
            data = fetch_eps_single_ticker(ticker)
            df = parse_quarterly_eps(data)
            all_eps_data.append(df)
        except RuntimeError as e:
            errors.append({
                "ticker": ticker,
                "error": str(e)
            })
            continue

    if all_eps_data:
        eps_df = pd.concat(all_eps_data, ignore_index=True)
    else:
        eps_df = pd.DataFrame(
            columns=["symbol", "fiscal_date", "reported_date", "reported_eps", "estimated_eps", "surprise_percentage"]
        )

    eps_df.to_csv("data/all_eps_data.csv", index=False)
    print("Errors:", errors)
    return eps_df
