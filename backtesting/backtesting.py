# backtesting/backtesting.py
import numpy as np

from backtesting.testing_functions import (check_explosiveness_feature,
                                           check_timing_danger_connection_to_earnings_move_bucket,
                                           check_corr_of_features,
                                           check_timing_danger_score_metric,
                                           three_way_regime_test,
                                           breakwater_regime_test,
                                           volatility_only_regime_test,
                                           evaluate_high_risk_earnings_regime,
                                           comparing_regime_results_to_volatility_only,
                                           regime_confusion_metrics,
                                           check_timing_danger_train_test,
                                           yearly_oos_report)
from backtesting.GPT_GENERATED_FILES.CHATGENERATED_stage5 import stage5_part_a, stage5_part_b
from backtesting.GPT_GENERATED_FILES.year_by_year_regime_eval import run_regime_eval 
from backtesting.GPT_GENERATED_FILES.year_by_year_regular import main as year_by_year_main
    
def backtesting_suite(backtesting_df):
    df = backtesting_df.copy()
    print("check_explosiveness_feature:")
    check_explosiveness_feature(df)
    print("-------------------------------\n")
    check_timing_danger_connection_to_earnings_move_bucket(df)
    print("-------------------------------\n")
    check_corr_of_features(df)
    print("-------------------------------\n")
    check_timing_danger_score_metric(df)
    print("-------------------------------\n")
    three_way_regime_test(df)
    print("-------------------------------\n")
    volatility_only_regime_test(df)
    print("-------------------------------\n")
    evaluate_high_risk_earnings_regime(df)
    print("-------------------------------\n")
    comparing_regime_results_to_volatility_only(df)
    print("-------------------------------\n")
    regime_confusion_metrics(df)
    print("-------------------------------\n")
    check_timing_danger_train_test(df)
    print("-------------------------------\n")
    yearly_oos_report(df)
    # stage5_df = stage5_part_a(backtesting_df)
    # stage5_b_df = stage5_part_b(backtesting_df)
    # stage5_year_by_year_df = year_by_year_main(backtesting_df)
    # stage5_year_by_year_holds_thresh = run_regime_eval(
    #     backtesting_df, mode="forward",
    #     train_years="2005-2010", test_years="2011-2025",
    #     report_train=True
    # )
    # print(stage5_year_by_year_holds_thresh[
    #     ["split","year","N_earnings","baseline_extreme_rate",
    #     "n_regime","regime_extreme_rate","lift", 
    #     "regime_share_of_events","regime_capture_of_extremes"]
    # ].to_string(index=False))




    