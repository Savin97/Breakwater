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
    os.makedirs("db/db_output", exist_ok=True)
    # ingest_all_stocks()
    # ingest_all_earnings_dates()
    #test_db() 


 