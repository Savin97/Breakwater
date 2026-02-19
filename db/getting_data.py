# db/getting_data.py
import os
import duckdb
import time
import pandas as pd
from datetime import datetime

from db.auxilary_functions import (create_prices_if_not_exists,
                                   create_earnings_table_if_not_exists,
                                   stock_already_in_db,
                                   fetch_full_daily_adjusted,
                                   fetch_recent_daily_adjusted,
                                   get_max_dates_by_stock)
from data_ingestion.api_functions import (get_earnings_data_from_api)
from data_utilities.formatting import to_float_or_none
from data_utilities.helper_funcs import get_alpha_vantage_api_key

from config import (DB_PATH,
                    STOCKS_START_DATE,
                    STOCK_LIST_PATH,
                    ALPHAVANTAGE_CALLS_PER_MINUTE)

def ingest_all_stocks():
    already = inserted = failed = 0
    FAILED_LOG_PATH = "db/db_output/failed_price_ingestion.txt"
    API_KEY = get_alpha_vantage_api_key()
    min_sleep = 60.0 / float(ALPHAVANTAGE_CALLS_PER_MINUTE)
    stocks = pd.read_csv(STOCK_LIST_PATH)["stock"].astype(str).str.strip().tolist()
    cutoff = pd.to_datetime(STOCKS_START_DATE).date()
    os.makedirs("db/db_output", exist_ok=True)

    if not API_KEY:
        raise RuntimeError("Set ALPHAVANTAGE_API_KEY env var first.")
    # reset failure log each run (simple)
    with open(FAILED_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("stock\terror\n")

    con = duckdb.connect(DB_PATH)
    # Ensure table exists with your schema
    create_prices_if_not_exists(con)

    global_max = con.execute("SELECT MAX(date) FROM prices").fetchone()[0] # type: ignore
    max_date_by_stock = get_max_dates_by_stock(con, "prices", "date")
    print("Prices DB global max date:", global_max)

    for i, stock in enumerate(stocks, start=1):
        # Block commented out 18/2/26
        # TODO: implement "stale-ticker" detection rule
        # if today_yyyy_mm_dd() - global_max > 10 trading days:
        #     stock = "inactive"
        #     # You could log something like:
        #     # DAY appears delisted (last price: 2026-02-03)
        #     # mark ticker as inactive
        #     # skip ingestion attempts
        #     continue   

        if stock_already_in_db(con, stock):
            already += 1
            stock_max_date = max_date_by_stock.get(stock)
            
            if i % 50 == 0:
                print(f"[{i}/{len(stocks)}] already in DB: {already}, inserted: {inserted}, failed: {failed}")
            data = fetch_recent_daily_adjusted(stock)
        else:
            data = fetch_full_daily_adjusted(stock)

        # parse JSON -> dataframe rows -> insert into prices

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
            if stock_max_date is not None:
                df = df[df["date"] > stock_max_date]   # only new days for that stock
            else:
                df = df[df["date"] >= cutoff]     # full ingest case
            if df.empty:
                # nothing new; update counters and move on
                already += 1
                time.sleep(min_sleep)
                continue

            df["ingested_at"] = datetime.now()

            con.register("tmp_prices", df)
            before = con.execute("SELECT COUNT(*) FROM prices WHERE stock = ?", [stock]).fetchone()[0] # type: ignore
            con.execute("INSERT OR IGNORE INTO prices SELECT * FROM tmp_prices;")
            con.unregister("tmp_prices")
            after  = con.execute("SELECT COUNT(*) FROM prices WHERE stock = ?", [stock]).fetchone()[0] # type: ignore
            added = after - before
            min_date, max_date = con.execute(
                "SELECT MIN(date), MAX(date) FROM prices WHERE stock = ?;",
                [stock]
            ).fetchone() # type: ignore
            print(f"Added {added} rows ({min_date} -> {max_date})")
            if added > 0:
                inserted += 1
            print(f"Inserted {after} rows ({min_date} -> {max_date})")
            # update cached max so later logic in same run is consistent
            max_date_by_stock[stock] = max_date
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
    print("\nIngesting Prices Done.")
    print("skipped/up-to-date:", already)
    print("inserted new:", inserted)
    print("failed:", failed)
    print("failures saved to:", FAILED_LOG_PATH)
    return 

def ingest_all_earnings_dates():
    already = inserted = failed = 0
    FAILED_EARNINGS_LOG_PATH = "db/db_output/failed_earnings_ingestion.txt"    
    API_KEY = get_alpha_vantage_api_key()
    min_sleep = 60.0 / float(ALPHAVANTAGE_CALLS_PER_MINUTE)
    stocks = pd.read_csv(STOCK_LIST_PATH)["stock"].astype(str).str.strip().tolist()
    cutoff = pd.to_datetime(STOCKS_START_DATE).date()
    os.makedirs("db/db_output", exist_ok=True)

    if not API_KEY:
        raise RuntimeError("Set ALPHAVANTAGE_API_KEY env var first.")
    # reset failure log each run (simple)
    with open(FAILED_EARNINGS_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("stock\terror\n")

    con = duckdb.connect(DB_PATH)
    # Ensure table exists with your schema
    create_earnings_table_if_not_exists(con)

    global_max_earnings_date = con.execute("SELECT MAX(earnings_date) FROM earnings").fetchone()[0] # type: ignore
    max_earnings_date_by_stock = get_max_dates_by_stock(con, "prices", "earnings_date")
    print("Earnings DB global max earnings_date:", global_max_earnings_date)

    for i, stock in enumerate(stocks, start=1):        
        if stock_already_in_db(con, stock):
            already += 1
            if i % 50 == 0:
                print(f"[{i}/{len(stocks)}] already in DB: {already}, inserted: {inserted}, failed: {failed}")
            data = get_earnings_data_from_api(stock, "compact")
        else:
            data = get_earnings_data_from_api(stock, "full")

        print(f"[{i}/{len(stocks)}] Fetching earnings data for {stock}...")
        try:
            if "quarterlyEarnings" not in data:
                raise RuntimeError(f"Bad payload keys: {list(data.keys())}. Snippet: {str(data)[:180]}")
            quarterly_earnings = data["quarterlyEarnings"]
            table_cols = ["stock", "reportedDate", "fiscalDateEnding", "reportedEPS", "estimatedEPS", "surprisePercentage"]
            rows = []   
            
            for quarter in quarterly_earnings:
                rows.append((stock, 
                             quarter["reportedDate"], 
                             quarter["fiscalDateEnding"], 
                             to_float_or_none(quarter["reportedEPS"]), 
                             to_float_or_none(quarter["estimatedEPS"]), 
                             to_float_or_none(quarter["surprisePercentage"])) )
            
            df = pd.DataFrame(rows, columns=table_cols)
            df = df.rename(columns={
                "reportedDate": "earnings_date",
                "fiscalDateEnding": "fiscal_end_date",
                "reportedEPS": "reported_eps",
                "estimatedEPS": "estimated_eps",
                "surprisePercentage": "surprise_percentage"
            })
            df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.date
            df["fiscal_end_date"] = pd.to_datetime(df["fiscal_end_date"]).dt.date
            
            df = df[df["earnings_date"] >= cutoff]
            df = df[df["fiscal_end_date"] >= cutoff]    # full ingest case
            df["surprise_percentage"] = df["surprise_percentage"] / 100
            df["ingested_at"] = datetime.now()

            count_before = con.execute("SELECT COUNT(*) FROM earnings WHERE stock = ?", [stock]).fetchone()[0] #type:ignore
            con.register("tmp_earnings_df", df)
            con.execute("INSERT OR IGNORE INTO earnings SELECT * FROM tmp_earnings_df")
            con.unregister("tmp_earnings_df")
            count_after = con.execute("SELECT COUNT(*) FROM earnings WHERE stock = ?", [stock]).fetchone()[0]#type:ignore
            added = count_after - count_before

            min_date, max_date = con.execute("""SELECT MIN(earnings_date), MAX(earnings_date) FROM earnings WHERE stock = ?;""", [stock]).fetchone() #type:ignore
            print(f"Added {added} rows ({min_date} -> {max_date})")
            if added != 0:
                inserted += 1
            print(f"Inserted {count_after} rows ({min_date} -> {max_date})")

        except Exception as e:
            failed += 1
            err = f"{type(e).__name__}: {e}"
            print(f"  FAILED {stock}: {err}")
            # ensure tmp view not left behind
            try:
                con.unregister("tmp_prices")
            except Exception:
                pass
            with open(FAILED_EARNINGS_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"{stock}\t{err}\n")
        # always sleep a bit to respect rate limits
        time.sleep(min_sleep)
        
    con.close()
    print("\nIngesting Earnings Done.")
    print("already in DB:", already)
    print("inserted new:", inserted)
    print("failed:", failed)
    print("Failures saved to:", FAILED_EARNINGS_LOG_PATH)

