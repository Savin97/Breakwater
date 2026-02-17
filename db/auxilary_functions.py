# db/auxilary_functions.py
import pandas as pd
import requests
import os
from pathlib import Path
from dotenv import load_dotenv
import duckdb


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

def create_earnings_date_if_not_exists(db_path):
    con = duckdb.connect(db_path)
    con.execute("""
                    CREATE TABLE IF NOT EXISTS earnings_dates (
                    stock TEXT,
                    earnings_date DATE,
                    fiscal_date_ending DATE,
                    ingested_at TIMESTAMP
                ); """)
    con.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS earnings_unique
        ON earnings_dates(stock, earnings_date, fiscal_date_ending);
    """)
    con.close()
    print("earnings_dates table ready.")

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

def fetch_recent_daily_adjusted(stock: str) -> dict:
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": stock,
        "outputsize": "compact",   # ← only change
        "apikey": API_KEY,
    }
    r = requests.get(ALPHAVANTAGE_BASE_URL, params=params, timeout=30)
    return r.json()


def test_db():
    con = duckdb.connect("data/breakwater.duckdb")
    print("\n\n---------------------\n")
    df = con.execute("""
        SELECT stock, COUNT(*) n, MIN(date) mind, MAX(date) maxd
        FROM prices
        GROUP BY stock
        ORDER BY stock
    """).df()   
    df.to_csv("count_db_test.csv",index=False)
    
    testing_if_all_fetched = con.execute("""
        WITH mx AS (SELECT MAX(date) AS global_max FROM prices)
        SELECT p.stock, MAX(p.date) AS max_date
        FROM prices p, mx
        GROUP BY p.stock, mx.global_max
        HAVING MAX(p.date) < mx.global_max
        ORDER BY max_date
        """).fetchdf()
    
    last_5_stocks = (con.execute(
        """
            WITH last_5_stocks AS (
            SELECT stock
            FROM prices
            GROUP BY stock
            ORDER BY MAX(ingested_at) DESC
            LIMIT 5
            )
            SELECT *
            FROM prices
            WHERE stock IN (SELECT stock FROM last_5_stocks)
            ORDER BY stock DESC, ingested_at DESC;
        """).fetchdf())

    print(testing_if_all_fetched.head())
    print(con.execute("SELECT COUNT(DISTINCT stock) FROM prices").fetchone())
    print(con.execute("SELECT DISTINCT stock FROM prices ORDER BY stock").fetchdf())
    print(con.execute("SELECT COUNT(*) FROM prices").fetchone())

    con.close()

    #print("rows:", len(df))
    #print("min/max:", df["date"].min(), df["date"].max())

def fetch_one_stock_into_db(db, TEST_STOCK = "AAPL"):
    HISTORY_START_DATE = "2000-01-01"
    connection = duckdb.connect(db)
    # 2) fetch 1 stock (full)
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": "AAPL",
        "outputsize": "full",
        "apikey": API_KEY,
    }
    r = requests.get(ALPHAVANTAGE_BASE_URL, params=params, timeout=30.0)
    data = r.json()
    if "Error Message" in data:
        raise ValueError ("API Error:", data["Error Message"])
    ts = data["Time Series (Daily)"] 
    
    # 3) parse to DataFrame (price = adjusted close)
    rows = []
    for date_str, ohlc in ts.items():
        price = float(ohlc["5. adjusted close"])
        rows.append(("AAPL", date_str, price, "10/2/26"))
    
    df = pd.DataFrame(rows, columns=["stock", "date", "price", "ingested_on"])
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # 4) keep only from 2000 onward
    df = df[df["date"] >= pd.to_datetime(HISTORY_START_DATE).date()]
    # 5) insert (simple: delete then insert for this stock)
    connection.execute("DELETE FROM prices WHERE stock = ?;", [TEST_STOCK])
    connection.register("tmp_prices", df)
    connection.execute("INSERT INTO prices SELECT * FROM tmp_prices;")
    connection.unregister("tmp_prices")

    # 6) verify
    n = connection.execute("SELECT COUNT(*) FROM prices WHERE stock = ?;", [TEST_STOCK]).fetchone()[0] # type:ignore
    mind = connection.execute("SELECT MIN(date), MAX(date) FROM prices WHERE stock = ?;", [TEST_STOCK]).fetchone()
    print("Inserted rows:", n)
    print("Min/Max date:", mind)