# pipeline/pipeline.py
from pipeline.stage1 import stage1
from pipeline.stage2 import stage2
from pipeline.stage3 import stage3
from pipeline.stage4 import stage4
from pipeline.stage5 import stage5
from pipeline.backtesting_stage import backtesting_stage
from pipeline.output import output_to_csv

def run_pipeline():
    """
        The pipeline stages:
        1. Fetch stock prices, earnings data, EPS data; Merge into a single DataFrame
        2. Engineer features
        3. Calculate Risk Score and Provide Explanations
        4. Back test

        NEW pipeline order:
        1. Build/Update DB
        2. Import from DB, merge and filter data( by start date, end date, stocks, etc)
        3. Engineer features
        4. Calculate Risk Score and Provide Explanations
        5. Report Generation
        - Side step: Backtesting
    """
    stage1()
    df = stage2() 
    feature_engineering = stage3(df)
    risk_scoring = stage4(feature_engineering)
    report = stage5(risk_scoring)
    return