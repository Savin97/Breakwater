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
    from back_testing.GPT_GENERATED_FILES.CHATGENERATED_stage5 import stage5_part_a, stage5_part_b
    from back_testing.GPT_GENERATED_FILES.year_by_year_regime_eval import run_regime_eval 
    #stage5_df = stage5_part_a(back_testing_df)
    #stage5_b_df = stage5_part_b(back_testing_df)
    #stage5_year_by_year_df = year_by_year_main(back_testing_df)
    stage5_year_by_year_holds_thresh = run_regime_eval(
        back_testing_df, mode="forward",
        train_years="2018-2021", test_years="2022-2025",
        report_train=True
    )
    print(stage5_year_by_year_holds_thresh[
        ["split","year","N_earnings","baseline_extreme_rate","n_regime","regime_extreme_rate","lift"]
    ].to_string(index=False))
    #output_to_csv(inputs_df, feature_engineering, scoring_df, back_testing_df, stage5_df)

    return 