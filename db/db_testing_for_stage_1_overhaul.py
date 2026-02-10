# db_testing_for_stage_1_overhaul.py
import json
import os
import duckdb
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

def main():
    ENV_PATH = Path(__file__).resolve().parents[1] / ".env"  # breakwater/.env
    # Load explicitly
    loaded = load_dotenv(dotenv_path=ENV_PATH, override=True)
    print("load_dotenv returned:", loaded)

    DB_PATH = "data/breakwater.duckdb"
    ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
    API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
    TEST_STOCK = "AAPL"
    HISTORY_START_DATE = "2000-01-01"  # store horizon
    BASE_DIR = Path(__file__).resolve().parents[1]  # breakwater/
    load_dotenv(BASE_DIR / ".env")

    # 1) connect + ensure table exists
    connection = duckdb.connect(DB_PATH)
    try:
        result = connection.execute("""
            CREATE TABLE IF NOT EXISTS prices(
                stock TEXT,
                date DATE,
                price DOUBLE,
                ingested_at TIMESTAMP
                );                
            """)
        
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
        n = connection.execute("SELECT COUNT(*) FROM prices WHERE stock = ?;", [TEST_STOCK]).fetchone()[0]
        mind = connection.execute("SELECT MIN(date), MAX(date) FROM prices WHERE stock = ?;", [TEST_STOCK]).fetchone()


    finally:
        connection.close()
    print("Inserted rows:", n)
    print("Min/Max date:", mind)

def read_db():
    DB_PATH = "data/breakwater.duckdb"

    con = duckdb.connect(DB_PATH)
    # Check table exists
    tables = con.execute("SHOW TABLES").fetchall()
    print("Tables:", tables)

    # Row count
    count = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    print("Total rows in prices:", count)

    # Sample rows
    sample = con.execute("""
        SELECT *
        FROM prices
        ORDER BY stock, date
        LIMIT 10
    """).df()
    
    print("\nSample rows:")
    print(sample)

    # AAPL stats
    aapl_stats = con.execute("""
        SELECT
            COUNT(*) AS n_rows,
            MIN(date) AS min_date,
            MAX(date) AS max_date
        FROM prices
        WHERE stock = 'AAPL'
    """).df()

    print("\nAAPL coverage:")
    print(aapl_stats)

    df = con.execute("""
        SELECT stock, COUNT(*) AS n
        FROM prices
        GROUP BY stock
        ORDER BY n DESC
    """).df()

    print(df.head())

    con.close()

def test_db():
    con = duckdb.connect("data/breakwater.duckdb")
    df = con.execute("""
        SELECT stock, date, price
        FROM prices 
        WHERE stock = 'AAPL'
        AND date >= DATE '2020-01-01'
        AND date <= DATE '2020-03-01'             
        ORDER BY stock, date
    """).df()

    check = con.execute("""
        SELECT (*) FROM prices WHERE stock = 'AAPL'
                        ORDER BY stock, date
        """).df()

    con.close()
    print(df.head())
    print("rows:", len(df))
    print("min/max:", df["date"].min(), df["date"].max())

    check.to_csv("check.csv", index=False)
    print("rows:", len(check))


if __name__ == "__main__":
    main()
    read_db()
    test_db()
