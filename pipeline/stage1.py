# pipeline/stage1.py
import duckdb
import warnings

from db.auxilary_functions import create_prices_table_if_not_exists, create_earnings_table_if_not_exists,create_sectors_table_if_not_exists, test_db
from db.getting_data import ingest_all_stocks, ingest_all_earnings_dates
from data_utilities.helper_funcs import directory_checks
from config import DB_PATH

def stage1():
    """
        Building / Updating DB
        1. Create DB/Make sure it exists.
        2. Create prices, earnings, sector tables / make sure they exist.
        3. Update tables or choose to leave them as-is (introduce a switch for this)
    """
    directory_checks()   
    warnings.filterwarnings('ignore')
    con = duckdb.connect(DB_PATH)
    create_prices_table_if_not_exists(con)
    create_earnings_table_if_not_exists(con)
    create_sectors_table_if_not_exists(con)
    # ingest_all_stocks()
    # ingest_all_earnings_dates()
    # test_db(con)
    return 

    # TODO: check to add for corrupted "ingested_at" values
    # if (df_to_insert["ingested_at"] < "2000-01-01").any():
    #     bad = df_to_insert[df_to_insert["ingested_at"] < "2000-01-01"].head(5)
    #   raise RuntimeError(f"Bad ingested_at about to be inserted:\n{bad}")