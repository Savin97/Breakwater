# pipeline/pipeline.py

from pipeline.stage1 import stage1
from pipeline.stage2 import stage2
from pipeline.stage3 import stage3
from pipeline.stage4 import stage4
from pipeline.stage5 import stage5
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
    risk_scoring = stage3(feature_engineering)
    backtesting = stage4(risk_scoring)
    report = stage5(risk_scoring)
    #output_to_csv(inputs_df, feature_engineering, risk_scoring, backtesting)

    return report