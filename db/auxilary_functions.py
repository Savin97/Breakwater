import requests
import os
from pathlib import Path
from dotenv import load_dotenv


ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"  # breakwater/.env
loaded = load_dotenv(dotenv_path=ENV_PATH, override=True)
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

def create_prices_if_not_exists(con):
    # ensure table exists (match your schema)
    con.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            stock TEXT,
            date  DATE,
            price DOUBLE,
            ingested_at TIMESTAMP
        );
    """)

def stock_already_in_db(con, stock: str) -> bool:
    n = con.execute("SELECT COUNT(*) FROM prices WHERE stock = ?;", [stock]).fetchone()[0]
    return n > 0


def fetch_full_daily_adjusted(stock: str) -> dict:
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": stock,
        "outputsize": "full",
        "apikey": API_KEY,
    }
    r = requests.get(ALPHAVANTAGE_BASE_URL, params=params, timeout=30)
    return r.json()