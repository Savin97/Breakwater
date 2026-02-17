# db/getting_data.py
import os
import duckdb
import time
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

from db.auxilary_functions import (create_prices_if_not_exists,
                                create_earnings_date_if_not_exists,
                                stock_already_in_db,
                                fetch_full_daily_adjusted,
                                fetch_recent_daily_adjusted)
from data_ingestion.api_functions import get_earnings_dates_from_api
from config import (BACKOFF_SECONDS,
                    MAX_RETRIES)

DB_PATH = "data/breakwater.duckdb"
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
TEST_STOCK = "AAPL"
HISTORY_START_DATE = "2000-01-01"  # store horizon
STOCK_LIST_PATH = "data/stock_list.csv"
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"  # breakwater/.env
loaded = load_dotenv(dotenv_path=ENV_PATH, override=True)
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
# Load explicitly
print("load_dotenv returned:", loaded)
BASE_DIR = Path(__file__).resolve().parents[1]  # breakwater/
load_dotenv(BASE_DIR / ".env")
CALLS_PER_MINUTE = 75

def ingest_all_stocks():
    
    FAILED_LOG_PATH = "db/db_output/failed_price_ingestion.txt"
    if not API_KEY:
        raise RuntimeError("Set ALPHAVANTAGE_API_KEY env var first.")

    os.makedirs("db/db_output", exist_ok=True)
    con = duckdb.connect(DB_PATH)

    # Ensure table exists with your schema
    create_prices_if_not_exists(con)
    # uniqueness constraint - only one row per (stock, date) pair, no duplicates allowed
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS prices_stock_date_uq ON prices(stock, date)")
    stocks = pd.read_csv(STOCK_LIST_PATH)["stock"].astype(str).str.strip().tolist()

    cutoff = pd.to_datetime(HISTORY_START_DATE).date()
    min_sleep = 60.0 / float(CALLS_PER_MINUTE)

    already = 0
    inserted = 0
    failed = 0

    # reset failure log each run (simple)
    with open(FAILED_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("stock\terror\n")
    global_max = con.execute("SELECT MAX(date) FROM prices").fetchone()[0] # type: ignore
    print("DB global max date:", global_max)

    for i, stock in enumerate(stocks, start=1):
         # skip stocks already at global max
        stock_max = con.execute(
            "SELECT MAX(date) FROM prices WHERE stock = ?",
            [stock]
        ).fetchone()[0] # type: ignore

        # If stock has no data yet, do full ingest
        if stock_max is None:
            data = fetch_full_daily_adjusted(stock)
        else:
            # If stock already caught up, skip API call completely
            if stock_max >= global_max:
                already += 1
                continue
            # TODO: implement "stale-ticker" detection rule
            # elif today_yyyy_mm_dd() - global_max > 10 trading days:
            #     stock = "inactive"
            #     # You could log something like:
            #     # DAY appears delisted (last price: 2026-02-03)
            #     # mark ticker as inactive
            #     # skip ingestion attempts
            #     continue   
            data = fetch_recent_daily_adjusted(stock)

        if stock_already_in_db(con, stock):
            data = fetch_recent_daily_adjusted(stock)
        else:
            data = fetch_full_daily_adjusted(stock)

        # parse JSON -> dataframe rows -> insert into prices

        if stock_already_in_db(con, stock):
            already += 1
            if i % 50 == 0:
                print(f"[{i}/{len(stocks)}] already in DB: {already}, inserted: {inserted}, failed: {failed}")

        print(f"[{i}/{len(stocks)}] Fetching {stock} ...")
        try:
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
            if stock_max is not None:
                df = df[df["date"] > stock_max]   # only new days for that stock
            else:
                df = df[df["date"] >= cutoff]     # full ingest case

            df["ingested_at"] = datetime.now()

            con.register("tmp_prices", df)
            before = con.execute("SELECT COUNT(*) FROM prices WHERE stock = ?", [stock]).fetchone()[0] # type: ignore
            con.execute("INSERT OR IGNORE INTO prices SELECT * FROM tmp_prices;")
            con.unregister("tmp_prices")
            after  = con.execute("SELECT COUNT(*) FROM prices WHERE stock = ?", [stock]).fetchone()[0] # type: ignore
            added = after - before
            n = con.execute("SELECT COUNT(*) FROM prices WHERE stock = ? ;", [stock]).fetchone()[0] # type: ignore
            mind, maxd = con.execute(
                "SELECT MIN(date), MAX(date) FROM prices WHERE stock = ?;",
                [stock]
            ).fetchone() # type: ignore
            print(f"Added {added} rows ({mind} → {maxd})")
            if added != 0:
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

def ingest_all_earnings_dates():
    FAILED_EARNINGS_LOG_PATH = "db/db_output/failed_earnings_ingestion.txt"
    with open(FAILED_EARNINGS_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("stock\terror\n")
    con = duckdb.connect(DB_PATH)
    create_earnings_date_if_not_exists(DB_PATH)

    stocks = pd.read_csv(STOCK_LIST_PATH)["stock"].astype(str).str.strip().tolist()
    min_sleep = 60.0 / float(CALLS_PER_MINUTE)
    inserted = 0
    failed = 0
    for i, stock in enumerate(stocks, start=1):
        print(f"[{i}/{len(stocks)}] Fetching earnings dates for {stock}...")
        q_earnings = get_stock_data(stock, con)
        print(q_earnings)
        break

    con.close()
    print("\nDone.")
    print("Tickers upserted:", inserted)
    print("Failed:", failed)
    print("Failures saved to:", FAILED_EARNINGS_LOG_PATH)

def get_stock_data(stock: str, con):
    try:
        data = get_earnings_dates_from_api(stock)
    except Exception as exc:
        last_err = exc
        raise RuntimeError(f"Alpha Vantage request failed for {stock}: \n{exc}")
    # most recent earnings_date we currently have for stock
    stock_max = con.execute(
        "SELECT MAX(earnings_date) FROM earnings_dates WHERE stock = ?",
        [stock]
    ).fetchone()[0]
    try:
        q_earnings = data["quarterlyEarnings"]
        return q_earnings
    except Exception as e:
        raise ValueError(f"Accessing quarterlyEarnings raised an exception {e}")
    







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

            n = con.execute("SELECT COUNT(*) FROM prices WHERE stock = ?;", [stock]).fetchone()[0] # type: ignore
            
            min_date, max_date = con.execute(
                "SELECT MIN(date), MAX(date) FROM prices WHERE stock = ?;",
                [stock]
            ).fetchone() # type: ignore

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


