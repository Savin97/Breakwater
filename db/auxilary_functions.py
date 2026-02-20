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
    # uniqueness constraint - only one row per (stock, date) pair, no duplicates allowed
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS prices_stock_date_uq ON prices(stock, date)")
    

def create_earnings_table_if_not_exists(con):
    con.execute("""
                    CREATE TABLE IF NOT EXISTS earnings (
                    stock TEXT,
                    earnings_date DATE,
                    fiscal_end_date DATE,
                    reported_eps DOUBLE,
                    estimated_eps DOUBLE,
                    surprise_percentage DOUBLE,
                    ingested_at TIMESTAMP
                ); """)
    con.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS earnings_unique
        ON earnings(stock, earnings_date, fiscal_end_date);
    """)
    print("earnings table ready.")

def stock_already_in_prices_db(con, stock: str) -> bool:
    n = con.execute("SELECT COUNT(*) FROM prices WHERE stock = ?;", [stock]).fetchone()[0]
    return n > 0

def stock_already_in_earnings_db(con, stock: str) -> bool:
    n = con.execute("SELECT 1 FROM earnings WHERE stock = ? LIMIT 1", [stock]).fetchone() is not None
    return n > 0


def fetch_daily_adjusted(stock: str, outputsize = "full") -> dict:
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": stock,
        "outputsize": outputsize,
        "apikey": API_KEY,
    }
    r = requests.get(ALPHAVANTAGE_BASE_URL, params=params, timeout=30)
    return r.json()

def test_db():
    con = duckdb.connect("data/breakwater.duckdb")
    print("\n\n---------------------\n")

    # Describe all tables
    print(con.execute("""SELECT
                    table_name,
                    column_name,
                    data_type
                    FROM information_schema.columns
                    ORDER BY table_name, ordinal_position; """).fetchall())
    con.execute("""SELECT *
                        FROM earnings
                        ORDER BY (stock,earnings_date); """).fetch_df().to_csv("db_df.csv",index=False)
    df = con.execute("""
        SELECT stock, COUNT(*) n, MIN(date) mind, MAX(date) maxd
        FROM prices
        GROUP BY stock
        ORDER BY stock
    """).df()   
    
    df.to_csv("count_db_test.csv",index=False)
    print("created test db in count_db_test.csv")
    
    testing_if_all_fetched = con.execute("""
        WITH mx AS (SELECT MAX(date) AS global_max FROM prices)
        SELECT p.stock, MAX(p.date) AS max_date
        FROM prices p, mx
        GROUP BY p.stock, mx.global_max
        HAVING MAX(p.date) < mx.global_max
        ORDER BY max_date
        """).fetchdf()

    print(testing_if_all_fetched.head())
    print(con.execute("SELECT COUNT(DISTINCT stock) FROM prices").fetchone())
    print(con.execute("SELECT DISTINCT stock FROM prices ORDER BY stock").fetchdf())
    print(con.execute("SELECT COUNT(*) FROM prices").fetchone())

    print("\n\n---------------------\nEarnings Table\n")
    df = con.execute("""
        SELECT stock, COUNT(*) n, MIN(earnings_date) mind, MAX(earnings_date) maxd
        FROM earnings
        GROUP BY stock
        ORDER BY stock
    """).df()   
    df.to_csv("count_db_test.csv",index=False)
    
    testing_if_all_fetched = con.execute("""
        WITH mx AS (SELECT MAX(earnings_date) AS global_max FROM earnings)
        SELECT e.stock, MAX(e.earnings_date) AS max_earnings_date
        FROM earnings e, mx
        GROUP BY e.stock, mx.global_max
        HAVING MAX(e.earnings_date) < mx.global_max
        ORDER BY max_earnings_date
        """).fetchdf()

    print(testing_if_all_fetched.head())
    print("Number of unique stocks in earnings: ", con.execute("SELECT COUNT(DISTINCT stock) FROM earnings").fetchone())
    print(con.execute("SELECT DISTINCT stock FROM earnings ORDER BY stock").fetchdf())
    print("Number of rows in earnings: ", con.execute("SELECT COUNT(*) FROM earnings").fetchone())

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

def get_max_dates_by_stock(con, table: str, date_col: str) -> dict[str, object]:
    rows = con.execute(f"""
        SELECT stock, MAX({date_col}) AS max_date
        FROM {table}
        GROUP BY stock
    """).fetchall()
    return {stock: max_date for stock, max_date in rows}