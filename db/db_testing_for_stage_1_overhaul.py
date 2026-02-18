# db_testing_for_stage_1_overhaul.py
import duckdb
import os
from pathlib import Path

from db.getting_data import (ingest_all_stocks,
                             ingest_all_earnings_dates)
from db.auxilary_functions import (test_db)

DB_PATH = "data/breakwater.duckdb"

def db_main():
    # connect + ensure table exists
    con = duckdb.connect(DB_PATH)
    try:
        os.makedirs("db/db_output", exist_ok=True)
        #ingest_all_stocks()
        #ingest_all_earnings_dates()
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
        test_db() 
    finally:
        con.close()


 