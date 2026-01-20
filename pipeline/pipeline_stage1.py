# pipeline_stage1
"""
    First stage of the pipeline.
    Data Ingestion
"""
import warnings
from pathlib import Path

from config import (MAX_RETRIES,
                    TICKERS_START_DATE,
                    BACKOFF_SECONDS,
                    TIMEOUT_SECONDS,
                    CORRECT_STOCK_COL_NAME,
                    LIST_OF_POSSIBLE_STOCK_COL_NAMES,
                    DEFAULT_FETCH_CHUNK_SIZE,
                    TICKERS_FILE_PATH
                    )
from data_utilities.formatting import today_yyyy_mm_dd, parse_date, change_column_name
from data_ingestion.fetch_stock_prices import fetch_stock_prices
from data_ingestion.fetch_earnings import fetch_earnings_dates
from data_utilities.clean_input import read_tickers_to_fetch
from data_utilities.merging import merge_prices_earnings_dates

# def stage1(tickers_path: str,
#             provider: str = "yfinance",
#             start: str = TICKERS_START_DATE,
#             end: str = today_yyyy_mm_dd(),
#             out: str = "data/prices_adj_close.parquet",
#             chunk_size: int = 50,
#             max_retries: int = MAX_RETRIES,
#             base_backoff_sec: float = BACKOFF_SECONDS,
#             timeout_sec: float = TIMEOUT_SECONDS
#         ):
def stage1(
        provider: str = "yfinance"
    ):
    warnings.filterwarnings('ignore')

    stock_prices = fetch_stock_prices(
        provider=provider,
        tickers_path=TICKERS_FILE_PATH,
        start=TICKERS_START_DATE,
        end=today_yyyy_mm_dd(),
        out = "data/prices_adj_close.csv",
        chunk_size=DEFAULT_FETCH_CHUNK_SIZE,
        max_retries=MAX_RETRIES,
        base_backoff_sec=BACKOFF_SECONDS,
        timeout_sec=TIMEOUT_SECONDS,
    )
    #stock_prices = pd.read_csv("data/prices_adj_close.csv")

    tickers = read_tickers_to_fetch(Path(TICKERS_FILE_PATH))
    earnings_dates = fetch_earnings_dates(
        symbols = tickers
    )
    #earnings_dates = pd.read_csv("data/earnings_dates.csv")

    stock_prices = change_column_name(stock_prices, LIST_OF_POSSIBLE_STOCK_COL_NAMES, CORRECT_STOCK_COL_NAME )
    earnings_dates = change_column_name(earnings_dates, LIST_OF_POSSIBLE_STOCK_COL_NAMES, CORRECT_STOCK_COL_NAME )
    # Sort to prep for merge_asof
    stock_prices["date"] = parse_date(stock_prices["date"])
    earnings_dates["earnings_date"] = parse_date(earnings_dates["earnings_date"])
    stock_prices = stock_prices.sort_values("date")
    earnings_dates = earnings_dates.sort_values("earnings_date")
    df = merge_prices_earnings_dates(stock_prices, earnings_dates) # df that holds stock prices, earnings dates, EPS data merged
    
    df = df.sort_values(["stock", "date"])

    # eps_data = get_eps_for_tickers(tickers)

    return df
    