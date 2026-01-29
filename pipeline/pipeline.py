# pipeline/pipeline.py

from pipeline.stage1 import stage1
from pipeline.stage2 import stage2
from pipeline.stage3 import stage3

def run_pipeline():
    """
        The engine pipeline stages:
        1. Fetch stock prices, earnings data, EPS data.
        2. Merge into a single DataFrame
        3. Engineer features
    """
    
    inputs_df = stage1() 
    feature_engineering = stage2(inputs_df)
    scoring_df = stage3(feature_engineering)

    inputs_df.to_csv("output/stage_1_df.csv", index=False)
    print("\n----------------")
    print("Stage 1 DF created in: output/stage_1_df.csv")

    feature_engineering.to_csv("output/stage_2_df.csv", index=False)
    print("----------------")
    print("Stage 2 DF created in: output/stage_2_df.csv")

    scoring_df.to_csv("output/stage_3_df.csv", index=False)
    print("----------------")
    print("Stage 3 DF created in: output/stage_3_df.csv")
    print("----------------")
    return scoring_df