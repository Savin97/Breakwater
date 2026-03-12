# db/auxilary_functions.py
def create_prices_table_if_not_exists(con):
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
    print("Prices table ready.")

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
    print("Earnings table ready.")

def create_sectors_data_table_if_not_exists(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS stock_data (
        stock TEXT PRIMARY KEY,
        company_name TEXT,
        sector TEXT,
        sub_sector TEXT,
        ingested_at TIMESTAMP
    ); """)
    print("Stock Data table ready.")

def stock_already_in_prices_db(con, stock: str) -> bool:
    n = con.execute("SELECT COUNT(*) FROM prices WHERE stock = ?;", [stock]).fetchone()[0]
    return n > 0

def get_max_dates_by_stock(con, table: str, date_col: str) -> dict[str, object]:
    rows = con.execute(f"""
        SELECT stock, MAX({date_col}) AS max_date
        FROM {table}
        GROUP BY stock
    """).fetchall()
    return {stock: max_date for stock, max_date in rows}

def test_db(con):
    print("\n\n---------------------\n")
    # Describe all tables
    print("Table description in the DB:")
    print(con.execute("""SELECT
                    table_name,
                    column_name,
                    data_type
                    FROM information_schema.columns
                    ORDER BY table_name, ordinal_position; """).fetchall())
    con.execute("""SELECT *
                        FROM earnings
                        ORDER BY stock,earnings_date; """).fetch_df().to_csv("earnings_db_df.csv",index=False)
    prices_count_df = con.execute("""
        SELECT stock, COUNT(*) n, MIN(date) mind, MAX(date) maxd
        FROM prices
        GROUP BY stock
        ORDER BY stock
    """).fetch_df()   
    prices_count_df.to_csv("count_prices_db_test.csv",index=False)
    print("\ncreated test db in count_db_test.csv\n")
    
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
    earnings_count_df = con.execute("""
        SELECT stock, COUNT(*) n, MIN(earnings_date) mind, MAX(earnings_date) maxd
        FROM earnings
        GROUP BY stock
        ORDER BY stock
    """).df()   
    earnings_count_df.to_csv("count_earnings_db_test.csv",index=False)
    print("\ncreated earnings_count_df.csv\n")
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
    