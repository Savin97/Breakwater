# pipeline_stage1
"""
    First stage of the pipeline.
    Data Ingestion
"""
import warnings

from config import TICKERS_START_DATE, TICKERS_END_DATE
from data_utilities.formatting import today_yyyy_mm_dd
from data_ingestion.fetch_stock_prices import fetch_stock_prices

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
    # def parse_args():
    #     p = argparse.ArgumentParser()
    #     p.add_argument("--provider", default="yfinance", choices=["yfinance"])
    #     p.add_argument("--tickers", required=True, help="tickers file: txt (one/line) or csv (symbol column)")
    #     p.add_argument("--start", default=TICKERS_START_DATE)
    #     p.add_argument("--end", default=today_yyyy_mm_dd())
    #     p.add_argument("--out", default="data/prices_adj_close.parquet")
    #     p.add_argument("--chunk-size", type=int, default=50)
    #     p.add_argument("--max-retries", type=int, default=5)
    #     p.add_argument("--base-backoff-sec", type=float, default=2.0)
    #     p.add_argument("--throttle-sec", type=float, default=0.5, help="sleep between batches to reduce blocks")
    #     return p.parse_args()

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
    
    #args = parse_args()
    # stock_prices = fetch_stock_prices(
    #     provider=args.provider,
    #     tickers_path=args.tickers,
    #     start=args.start,
    #     end=args.end,
    #     out=args.out,
    #     chunk_size=args.chunk_size,
    #     max_retries=args.max_retries,
    #     base_backoff_sec=args.base_backoff_sec,
    #     throttle_sec=args.throttle_sec,
    # )

    return stock_prices
    