from pathlib import Path
from pipeline.pipeline_stage1 import stage1

def run_pipeline():
    tickers_path = Path("stock_tickers_file.txt")
    print("Running the pipeline...\n----------------\n")
    prices_and_earnings_df = stage1(tickers_path) 

    prices_and_earnings_df.to_csv("output/prices.csv", index=False)
    return prices_and_earnings_df