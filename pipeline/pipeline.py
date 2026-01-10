from pathlib import Path
from pipeline.pipeline_stage1 import stage1

def run_pipeline():
    tickers_path = Path("stock_tickers_file.txt")
    print("Running the pipeline...\n----------------\n")
    df_prices = stage1(tickers_path) 
    df_prices.to_csv("output/prices.csv", index=False)
    return df_prices