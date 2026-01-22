# pipeline_stage1
"""
    First stage of the pipeline.
    Data Ingestion
"""
import warnings
from pathlib import Path

from config import ( CORRECT_STOCK_COL_NAME,
                    LIST_OF_POSSIBLE_STOCK_COL_NAMES,
                    TICKERS_FILE_PATH )
from data_utilities.formatting import today_yyyy_mm_dd, parse_date, change_column_name
from data_ingestion.fetch_stock_prices import fetch_stock_prices
from data_ingestion.fetch_earnings import fetch_earnings_dates
from data_ingestion.fetch_eps import fetch_eps
from data_ingestion.fetch_sectors import fetch_sectors
from data_utilities.clean_input import read_tickers_to_fetch
from data_utilities.merging import merge_prices_earnings_dates, merge_main_df_with_eps_df

# def stage1(tickers_path: str,
#             provider: str = "yfinance",
#             start: str = TICKERS_START_DATE,
#             end: str = today_yyyy_mm_dd(),
#             out: str = "data/stock_prices.parquet",
#             chunk_size: int = 50,
#             max_retries: int = MAX_RETRIES,
#             base_backoff_sec: float = BACKOFF_SECONDS,
#             timeout_sec: float = TIMEOUT_SECONDS
#         ):

def stage1( provider: str = "yfinance" ):

    warnings.filterwarnings('ignore')
    tickers = read_tickers_to_fetch(Path(TICKERS_FILE_PATH))
    print(f"{len(tickers)} Tickers to fetch.\n")
    
    stock_prices = fetch_stock_prices(provider=provider)
    earnings_dates = fetch_earnings_dates(symbols = tickers)

    stock_prices = change_column_name(stock_prices, LIST_OF_POSSIBLE_STOCK_COL_NAMES, CORRECT_STOCK_COL_NAME )
    earnings_dates = change_column_name(earnings_dates, LIST_OF_POSSIBLE_STOCK_COL_NAMES, CORRECT_STOCK_COL_NAME )
    # Sort to prep for merge_asof
    stock_prices["date"] = parse_date(stock_prices["date"])
    earnings_dates["earnings_date"] = parse_date(earnings_dates["earnings_date"])

    stock_prices = stock_prices.sort_values("date")
    earnings_dates = earnings_dates.sort_values("earnings_date")
    
    eps_data = fetch_eps(tickers)

    df = merge_prices_earnings_dates(stock_prices, earnings_dates) # df that holds stock prices, earnings dates, EPS data merged
    df = df.sort_values(["stock", "date"])
    df = merge_main_df_with_eps_df(df, eps_data)

    sector_map = fetch_sectors()

    return df
    