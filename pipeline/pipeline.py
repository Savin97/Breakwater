# pipeline/pipeline.py

from pipeline.stage1 import stage1
from pipeline.stage2 import stage2
from pipeline.stage3 import stage3
from pipeline.stage4 import stage4
from config import cols_to_drop_for_output

def run_pipeline():
    """
        The engine pipeline stages:
        1. Fetch stock prices, earnings data, EPS data; Merge into a single DataFrame
        2. Engineer features
        3. Calculate Risk Score and Provide Explanations
        4. Back-test
    """
    
    inputs_df = stage1() 
    feature_engineering = stage2(inputs_df)
    scoring_df = stage3(feature_engineering)
    back_testing_df = stage4(scoring_df)
    
    #inputs_df.to_csv("output/stage_1_df.csv", index=False)
    # print("----------------\nStage 1 DF created in: output/stage_1_df.csv")

    #feature_engineering.to_csv("output/stage_2_df.csv", index=False)
    # print("----------------\nStage 2 DF created in: output/stage_2_df.csv")

    scoring_df.to_csv("output/stage_3_df.csv", index=False)
    print("----------------\nStage 3 DF created in: output/stage_3_df.csv")

    # TODO: only keeping earnings days so far
    #back_testing_df = back_testing_df[back_testing_df["is_earnings_day"] == 1]

    #back_testing_df = back_testing_df.drop(columns=cols_to_drop_for_output)
    back_testing_df.to_csv("output/back_testing_df.csv", index=False)
    print("----------------\nBack testing DF created in: output/back_testing_df.csv")

    # output_df = scoring_df.drop(columns=cols_to_drop_for_output)
    # output_df.to_csv("output/output_df.csv", index=False)
    # print("----------------\noutput_df created in: output/output_df.csv", "\n----------------")

    return scoring_df