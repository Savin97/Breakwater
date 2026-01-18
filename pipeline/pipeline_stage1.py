# pipeline_stage1
"""
    First stage of the pipeline.
    Data Ingestion
"""
import warnings
import pandas as pd
from pathlib import Path

from config import (MAX_RETRIES,
                    TICKERS_START_DATE,
                      BACKOFF_SECONDS,
                        TIMEOUT_SECONDS,
                          CORRECT_STOCK_COL_NAME,
                          LIST_OF_POSSIBLE_STOCK_COL_NAMES
)
from data_ingestion.fetch_eps import get_eps_for_tickers
from data_utilities.formatting import today_yyyy_mm_dd, change_column_name
from data_ingestion.fetch_stock_prices import fetch_stock_prices
from data_ingestion.fetch_earnings import fetch_earnings_dates
from data_utilities.clean_input import read_tickers_to_fetch
from data_utilities.merge_input_data import merge_prices_earnings_dates

def stage1(tickers_path: str,
            provider: str = "yfinance",
            start: str = TICKERS_START_DATE,
            end: str = today_yyyy_mm_dd(),
            out: str = "data/prices_adj_close.parquet",
            chunk_size: int = 50,
            max_retries: int = MAX_RETRIES,
            base_backoff_sec: float = BACKOFF_SECONDS,
            timeout_sec: float = TIMEOUT_SECONDS
        ):
    warnings.filterwarnings('ignore')
    tickers = read_tickers_to_fetch(Path(tickers_path))

    # stock_prices = fetch_stock_prices(
    #     provider=provider,
    #     tickers_path=tickers_path,
    #     start=start,
    #     end=end,
    #     out=out,
    #     chunk_size=chunk_size,
    #     max_retries=max_retries,
    #     base_backoff_sec=base_backoff_sec,
    #     timeout_sec=timeout_sec,
    # )
    stock_prices = pd.read_csv("data/prices_adj_close.csv")

    # earnings_dates = fetch_earnings_dates(
    #     symbols = tickers
    # )
    earnings_dates = pd.read_csv("data/earnings_dates.csv")

    stock_prices = change_column_name(stock_prices, CORRECT_STOCK_COL_NAME, LIST_OF_POSSIBLE_STOCK_COL_NAMES)
    earnings_dates = change_column_name(earnings_dates, CORRECT_STOCK_COL_NAME, LIST_OF_POSSIBLE_STOCK_COL_NAMES)

    stock_prices = stock_prices.sort_values(["stock", "date"])
    earnings_dates = earnings_dates.sort_values(["stock", "earnings_date"])

    df = merge_prices_earnings_dates(stock_prices, earnings_dates) # df that holds stock prices, earnings dates, EPS data merged
    
    eps_data = get_eps_for_tickers(tickers)

    return df
    