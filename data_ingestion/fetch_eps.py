import requests
import os 
from config import ALPHAVANTAGE_BASE_URL

def fetch_eps():
    """
        Fetch EPS data for a list of tickers from Alpha Vantage.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        api_key = "demo"  # Use demo key for testing purposes
    tickers = ["AAPL", "MSFT", "GOOGL"]
    for ticker in tickers:
        url = f"{ALPHAVANTAGE_BASE_URL}?function=EARNINGS&symbol={ticker}&apikey={api_key}"
        r = requests.get(url)
        data = r.json()
        print(data)

if __name__ == "__main__":
    fetch_eps()