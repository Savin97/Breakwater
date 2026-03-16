# pipeline/pipeline.py
from pipeline.stage1 import stage1
from pipeline.stage2 import stage2
from pipeline.stage3 import stage3
from pipeline.stage4 import stage4
from pipeline.stage5 import stage5

def run_pipeline():
    """
        The pipeline stages:
        1. Build/Update DB
        2. Import from DB, merge and filter data (by start date, end date, stocks, etc)
        3. Engineer features
        4. Calculate Risk Score and Provide Explanations
        5. Report Generation
        - Side step: Backtesting
    """
    import warnings
    warnings.filterwarnings('ignore')
    # stage1()
    # df = stage2() 
    # feature_engineered_df = stage3(df)
    # risk_scored_df = stage4(feature_engineered_df)
    # report = stage5(risk_scored_df)
    stage5()
    #print(risk_scored_df.columns)
    #partial_df = risk_scored_df[risk_scored_df["date"] >= "2025-10-01"]
    #partial_df.to_csv("partial_df.csv",index=False)
    # return partial_df # For streamlit