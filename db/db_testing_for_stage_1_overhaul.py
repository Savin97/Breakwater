# db_testing_for_stage_1_overhaul.py
import duckdb
import os
from pathlib import Path

from db.getting_data import (ingest_all_stocks,
                             ingest_all_earnings_dates)
from db.auxilary_functions import (create_prices_if_not_exists,
                                   create_earnings_date_if_not_exists,
                                   test_db)
DB_PATH = "data/breakwater.duckdb"

def main():
    # connect + ensure table exists
    con = duckdb.connect(DB_PATH)
    try:
        os.makedirs("db/db_output", exist_ok=True)
        create_prices_if_not_exists(con)
        ingest_all_earnings_dates()
        # Describe all tables
        print(con.execute("""SELECT
                            table_name,
                            column_name,
                            data_type
                            FROM information_schema.columns
                            ORDER BY table_name, ordinal_position; """).fetchall())
    finally:
        con.close()

if __name__ == "__main__":
    main()

 