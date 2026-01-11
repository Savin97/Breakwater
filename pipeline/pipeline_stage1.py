# pipeline_stage1
"""
    First stage of the pipeline.
    Data Ingestion
"""
import warnings

from config import TICKERS_START_DATE, TICKERS_END_DATE
from data_utilities.formatting import today_yyyy_mm_dd
from data_ingestion.fetch_stock_prices import fetch_stock_prices
from data_ingestion.fetch_earnings import fetch_earnings_dates

def stage1(tickers_path: str,
            provider: str = "yfinance",
            start: str = TICKERS_START_DATE,
            end: str = today_yyyy_mm_dd(),
            out: str = "data/prices_adj_close.parquet",
            chunk_size: int = 50,
            max_retries: int = 5,
            base_backoff_sec: float = 2.0,
            throttle_sec: float = 0.5
        ):
    warnings.filterwarnings('ignore')

    stock_prices = fetch_stock_prices(
        provider=provider,
        tickers_path=tickers_path,
        start=start,
        end=end,
        out=out,
        chunk_size=chunk_size,
        max_retries=max_retries,
        base_backoff_sec=base_backoff_sec,
        throttle_sec=throttle_sec,
    )
    earnings_dates = fetch_earnings_dates(
        tickers_path = tickers_path,
        start_date = start
    )

    return stock_prices
    