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
    """
    import warnings
    warnings.filterwarnings('ignore')
    # from testing import testing_scores, features_test
    # features_test()

    stage1(update=False)
    df = stage2() 
    feature_engineered_df = stage3(df)
    risk_scored_df = stage4(feature_engineered_df)
    report = stage5(risk_scored_df)
