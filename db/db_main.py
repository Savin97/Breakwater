# db_main.py
import os

from db.getting_data import (ingest_all_stocks,
                             ingest_all_earnings_dates)
from db.auxilary_functions import (test_db)

def db_main():
    os.makedirs("db/db_output", exist_ok=True)
    ingest_all_earnings_dates()
    # ingest_all_stocks()
    test_db() 



 