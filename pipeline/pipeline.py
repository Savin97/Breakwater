from pathlib import Path
from pipeline.pipeline_stage1 import stage1

def run_pipeline():
    """
        The engine pipeline stages:
        1. Fetch stock prices, earnings data, EPS data.
        2. Merge into a single DataFrame
        3. Engineer features

    """

    prices_and_earnings_df = stage1() 
    prices_and_earnings_df.to_csv("output/df.csv", index=False)
    print("\n----------------\n")
    print("DF csv created in: output/df.csv")
    return prices_and_earnings_df