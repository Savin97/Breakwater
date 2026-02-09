# pipeline/pipeline.py

from pipeline.stage1 import stage1
from pipeline.stage2 import stage2
from pipeline.stage3 import stage3
from pipeline.stage4 import stage4
from pipeline.output import output_to_csv

def run_pipeline():
    """
        The pipeline stages:
        1. Fetch stock prices, earnings data, EPS data; Merge into a single DataFrame
        2. Engineer features
        3. Calculate Risk Score and Provide Explanations
        4. Back test
    """
    inputs_df = stage1() 
    feature_engineering = stage2(inputs_df)
    scoring_df = stage3(feature_engineering)
    back_testing_df = stage4(scoring_df)
    output_to_csv(inputs_df, feature_engineering, scoring_df, back_testing_df)

    return 