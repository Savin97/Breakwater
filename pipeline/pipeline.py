# pipeline/pipeline.py

from pipeline.stage1 import stage1
from pipeline.stage2 import stage2

def run_pipeline():
    """
        The engine pipeline stages:
        1. Fetch stock prices, earnings data, EPS data.
        2. Merge into a single DataFrame
        3. Engineer features
    """
    
    inputs_df = stage1() 
    feature_engineering = stage2(inputs_df)

    inputs_df.to_csv("output/stage_1_df.csv", index=False)
    feature_engineering.to_csv("output/stage_2_df.csv", index=False)
    print("\n----------------\n")
    print("Stage 1 DF created in: output/stage_2_df.csv")
    print("----------------\n")
    print("Stage 2 DF created in: output/stage_2_df.csv")
    print("----------------\n")
    return inputs_df