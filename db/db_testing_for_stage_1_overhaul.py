# db_testing_for_stage_1_overhaul.py
import json
import os
import duckdb
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

from auxilary_functions import (create_prices_if_not_exists,
                                stock_already_in_db,
                                fetch_full_daily_adjusted)

DB_PATH = "data/breakwater.duckdb"
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
TEST_STOCK = "AAPL"
HISTORY_START_DATE = "2000-01-01"  # store horizon
STOCK_LIST_PATH = "data/stock_list.csv"
HISTORY_START_DATE = "2000-01-01"
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"  # breakwater/.env
loaded = load_dotenv(dotenv_path=ENV_PATH, override=True)
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
# Load explicitly
print("load_dotenv returned:", loaded)
BASE_DIR = Path(__file__).resolve().parents[1]  # breakwater/
load_dotenv(BASE_DIR / ".env")
# how many new stocks to ingest this run
N_NEW_STOCKS = 5


def main():
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
    # print("Inserted rows:", n)
    # print("Min/Max date:", mind)

def test_db():
    con = duckdb.connect("data/breakwater.duckdb")
    df = con.execute("""
        SELECT *
        FROM prices             
        ORDER BY stock, date
    """).df()   

    print("\n---------------------\n")
    # df.to_csv("test.csv",index=False)
    print(con.execute("SELECT COUNT(DISTINCT stock) FROM prices").fetchone())
    print(con.execute("SELECT COUNT(*) FROM prices").fetchone())
    print(con.execute("""
    SELECT stock, COUNT(*) n, MIN(date) mind, MAX(date) maxd
    FROM prices
    GROUP BY stock
    ORDER BY stock
    """).df())
    con.close()

    #print("rows:", len(df))
    #print("min/max:", df["date"].min(), df["date"].max())

def ingesting_more_stocks():
    if not API_KEY:
        raise RuntimeError("Set ALPHAVANTAGE_API_KEY env var first.")
    CALLS_PER_MINUTE = 75
    # how many new stocks to ingest this run
    NUM_OF_NEW_STOCKS = 5
    con = duckdb.connect(DB_PATH)
    create_prices_if_not_exists(con)
    stocks = pd.read_csv(STOCK_LIST_PATH)["stock"].astype(str).str.strip().tolist()
    cutoff = pd.to_datetime(HISTORY_START_DATE).date()
    min_sleep = 60.0 / float(CALLS_PER_MINUTE)
    added = 0

    for stock in stocks:
        if stock_already_in_db(con, stock):
            continue
        print(f"Fetching {stock}")
        try:
            data = fetch_full_daily_adjusted(stock)
            if "Time Series (Daily)" not in data:
                raise RuntimeError(f"Bad payload for {stock}. Keys: {list(data.keys())}. Full: {data}")
            ts = data["Time Series (Daily)"]
            rows = []
            for date_str, ohlc in ts.items():
                price = float(ohlc["5. adjusted close"])
                rows.append((stock, date_str, price))
                
            df = pd.DataFrame(rows, columns=["stock", "date", "price"])
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df = df[df["date"] >= cutoff]
            df["ingested_at"] = datetime.now()

            # temp register, for insertion (no delete needed because we ensured stock wasn't in DB)
            con.register("tmp_prices", df)
            con.execute("INSERT INTO prices SELECT * FROM tmp_prices;")
            con.unregister("tmp_prices")

            n = con.execute("SELECT COUNT(*) FROM prices WHERE stock = ?;", [stock]).fetchone()[0]
            
            min_date, max_date = con.execute(
                "SELECT MIN(date), MAX(date) FROM prices WHERE stock = ?;",
                [stock]
            ).fetchone()

            print(f"  inserted {n} rows ({min_date} → {max_date})")

            added += 1
            if added >= NUM_OF_NEW_STOCKS:
                break

            time.sleep(min_sleep)
        except Exception as e:
            # report + continue
            print(f"  FAILED {stock}: {type(e).__name__}: {e}")

            # make sure temp view doesn't linger if failure happened after register
            try:
                con.unregister("tmp_prices")
            except Exception:
                pass

            # small sleep anyway (helps if it was rate-limit)
            time.sleep(min_sleep)
            continue

    con.close()
    print(f"Done. Added {added} new stocks.")

def ingest_all_stocks():
    HISTORY_START_DATE = "2000-01-01"
    CALLS_PER_MINUTE = 75
    FAILED_LOG_PATH = "db/db_output/failed_price_ingestion.txt"
    if not API_KEY:
        raise RuntimeError("Set ALPHAVANTAGE_API_KEY env var first.")

    os.makedirs("db/db_output", exist_ok=True)
    con = duckdb.connect(DB_PATH)

    # Ensure table exists with your schema
    create_prices_if_not_exists(con)
    
    stocks = pd.read_csv(STOCK_LIST_PATH)["stock"].astype(str).str.strip().tolist()

    cutoff = pd.to_datetime(HISTORY_START_DATE).date()
    min_sleep = 60.0 / float(CALLS_PER_MINUTE)

    already = 0
    inserted = 0
    failed = 0

    # reset failure log each run (simple)
    with open(FAILED_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("stock\terror\n")

    for i, stock in enumerate(stocks, start=1):
        if stock_already_in_db(con, stock):
            already += 1
            if i % 50 == 0:
                print(f"[{i}/{len(stocks)}] already in DB: {already}, inserted: {inserted}, failed: {failed}")
            continue

        print(f"[{i}/{len(stocks)}] Fetching {stock} ...")
        try:
            data = fetch_full_daily_adjusted(stock)

            if "Time Series (Daily)" not in data:
                # AlphaVantage commonly returns {"Note": "..."} on rate limit
                raise RuntimeError(f"Bad payload keys: {list(data.keys())}. Snippet: {str(data)[:180]}")

            ts = data["Time Series (Daily)"]

            rows = []
            for date_str, ohlc in ts.items():
                price = float(ohlc["5. adjusted close"])
                rows.append((stock, date_str, price))

            df = pd.DataFrame(rows, columns=["stock", "date", "price"])
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df = df[df["date"] >= cutoff]
            df["ingested_at"] = datetime.now()

            con.register("tmp_prices", df)
            con.execute("INSERT INTO prices SELECT * FROM tmp_prices;")
            con.unregister("tmp_prices")

            n = con.execute("SELECT COUNT(*) FROM prices WHERE stock = ? ;", [stock]).fetchone()[0] # type: ignore
            mind, maxd = con.execute(
                "SELECT MIN(date), MAX(date) FROM prices WHERE stock = ?;",
                [stock]
            ).fetchone() # type: ignore
            inserted += 1
            print(f"Inserted {n} rows ({mind} → {maxd})")

        except Exception as e:
            failed += 1
            err = f"{type(e).__name__}: {e}"
            print(f"  FAILED {stock}: {err}")

            # ensure tmp view not left behind
            try:
                con.unregister("tmp_prices")
            except Exception:
                pass

            with open(FAILED_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"{stock}\t{err}\n")

        # always sleep a bit to respect rate limits
        time.sleep(min_sleep)

    con.close()
    print("\nDONE")
    print("already in DB:", already)
    print("inserted new:", inserted)
    print("failed:", failed)
    print("failures saved to:", FAILED_LOG_PATH)
    return 



if __name__ == "__main__":


    main()
    #test_db()
    #ingesting_more_stocks()
    ingest_all_stocks()
