from pipeline.pipeline_stage1 import stage1

def run_pipeline():
    print("Running the pipeline...")
    df_prices = stage1(tickers_path="data/tickers.txt") 
    return df_prices