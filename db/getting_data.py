# db/getting_data.py
import os
import duckdb
import time
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, date

from db.auxilary_functions import (create_prices_table_if_not_exists,
                                   create_earnings_table_if_not_exists,
                                   stock_already_in_prices_db,
                                   fetch_daily_adjusted,
                                   get_max_dates_by_stock)
from data_ingestion.api_functions import (get_earnings_data_from_api)
from data_utilities.formatting import to_float_or_none, today_yyyy_mm_dd
from data_utilities.helper_funcs import get_alpha_vantage_api_key, read_stocks_to_fetch

from config import (DB_PATH,
                    STOCKS_START_DATE,
                    STOCK_LIST_PATH,
                    ALPHAVANTAGE_CALLS_PER_MINUTE,
                    BACKOFF_SECONDS)

def trading_days_since(last_date, calendar="NYSE"):
                cal = mcal.get_calendar(calendar)
                schedule = cal.schedule(start_date=last_date, end_date=date.today())
                return len(schedule)

def ingest_all_stocks():
    """
        This function updates the 'prices' table in the duckDB.
    """
    already, inserted, failed = 0,0,0
    FAILED_LOG_PATH = "debugging/failed_price_ingestion.txt"
    API_KEY = get_alpha_vantage_api_key()
    min_sleep = 60.0 / float(ALPHAVANTAGE_CALLS_PER_MINUTE)
    stocks = read_stocks_to_fetch()
    if not stocks:
        raise ValueError("No stocks found.")
    cutoff = pd.to_datetime(STOCKS_START_DATE).date()

    if not API_KEY:
        raise RuntimeError("Set ALPHAVANTAGE_API_KEY env var first.")
    # reset failure log each run (simple)
    with open(FAILED_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("stock\terror\n")

    con = duckdb.connect(DB_PATH)

    global_max_date = con.execute("SELECT MAX(date) FROM prices").fetchone()[0] # type: ignore
    max_date_by_stock = get_max_dates_by_stock(con, "prices", "date")

    print(f"Stocks: {len(stocks)}")
    print("Prices DB global max date:", global_max_date)

    for i, stock in enumerate(stocks, start=1):
        stock_max_date = max_date_by_stock.get(stock)
        print(f"\n{stock} Max date is {stock_max_date}")
        try:
            outputsize="full"
            if stock_already_in_prices_db(con, stock):
                if i % 50 == 0:
                    print(f"[{i}/{len(stocks)}] already in DB: {already}, inserted: {inserted}, failed: {failed}")
                # TODO: 18/2/26 implement "stale-ticker" detection rule
                # if trading_days_since(global_max_date) > 5: # >10 trading days
                #     with open(FAILED_LOG_PATH, "a", encoding="utf-8") as f:
                #         f.write(f"{stock}\t{"Appears to be delisted, (last price: {stock_max_date})"}\n")
                #     # You could log something like:
                #     # DAY appears delisted (last price: 2026-02-03)
                #     # stock = "inactive" # mark ticker as inactive
                #     continue # skip ingestion attempts
                if (stock_max_date is not None) and (global_max_date is not None) and (stock_max_date >= global_max_date):
                    already += 1
                    print(f"{stock} is up to date")
                    continue
                else:
                    outputsize = "compact"

            print(f"[{i}/{len(stocks)}] Fetching {stock} ...")
            data = fetch_daily_adjusted(stock, outputsize)
            # Basic failure handling (AlphaVantage often returns errors inside JSON)
            if "Error Message" in data:
                print(f"[{i}] {stock}: no data / error")
                time.sleep(min_sleep)
                continue

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
            print(f" FAILED {stock}: {err}")
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
    FAILED_EARNINGS_LOG_PATH = "debugging/failed_earnings_ingestion.txt"    
    API_KEY = get_alpha_vantage_api_key()
    min_sleep = 60.0 / float(ALPHAVANTAGE_CALLS_PER_MINUTE)
    stocks = pd.read_csv(STOCK_LIST_PATH)["stock"].astype(str).str.strip().tolist()
    cutoff = pd.to_datetime(STOCKS_START_DATE).date()

    if not API_KEY:
        raise RuntimeError("Set ALPHAVANTAGE_API_KEY env var first.")
    # reset failure log each run (simple)
    with open(FAILED_EARNINGS_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("stock\terror\n")

    con = duckdb.connect(DB_PATH)
    # Ensure table exists with your schema
    create_earnings_table_if_not_exists(con)

    # cache current max earnings_date per stock
    max_earnings_date_by_stock = get_max_dates_by_stock(con, "earnings", "earnings_date")

    # heuristic freshness window (quarterly): if you already have something in last 80 days, skip
    today = datetime.now().date()
    fresh_window_days = 0

    for i, stock in enumerate(stocks, start=1):  
        stock_earn_max_date = max_earnings_date_by_stock.get(stock)

        if stock_earn_max_date is not None and (today - stock_earn_max_date).days <= fresh_window_days: # type: ignore
            already += 1
            print(f"{stock} is up to date")
            if i % 50 == 0:
                print(f"[{i}/{len(stocks)}] skipped(fresh): {already}, inserted: {inserted}, failed: {failed}")
            continue

        data = get_earnings_data_from_api(stock)    
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
            df = df[df["fiscal_end_date"] >= cutoff] 

            if df.empty:
                already += 1
                time.sleep(min_sleep)
                continue

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

