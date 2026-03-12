# backtesting/backtesting.py
from backtesting.testing_functions import (
    check_explosiveness_feature,
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
    yearly_oos_report,
    forward_eval_onefactor,
    forward_eval_twofactor_and)
from backtesting.features_for_backtesting import (
    engineer_abs_reaction_p75_rolling,
    engineer_abs_reaction_p90_rolling)
from backtesting.GPT_GENERATED_FILES.CHATGENERATED_stage5 import stage5_part_a, stage5_part_b
from backtesting.GPT_GENERATED_FILES.year_by_year_regime_eval import run_regime_eval 
from backtesting.GPT_GENERATED_FILES.year_by_year_regular import main as year_by_year_main
from pipeline.pipeline import run_pipeline

def backtesting_suite(input_df):
    """    
        Backtesting Stage
        Produces credibility tables (calibration, lift, hit rates, bucket stats, stability by year/sector).

        You now have your first defensible rule:
        Tail risk increases materially when:
        timing_danger is high
        stock_vs_sector_vol ≥ 1
    """
    df = input_df.copy()
    print("Backtesting Stage...")
    print("-------------------------------\ncheck_explosiveness_feature:")
    check_explosiveness_feature(df)
    print("-------------------------------\ncheck_timing_danger_connection_to_earnings_move_bucket:")
    check_timing_danger_connection_to_earnings_move_bucket(df)
    print("-------------------------------\ncheck_corr_of_features:")
    check_corr_of_features(df)
    print("-------------------------------\ncheck_timing_danger_score_metric:")
    check_timing_danger_score_metric(df)
    print("-------------------------------\nthree_way_regime_test:")
    three_way_regime_test(df)
    print("-------------------------------\nVolatility Only:\n")
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

    # 1) Expanding prior only
    prior_stats, prior_thr = forward_eval_onefactor(df, "abs_reaction_p75", q=0.90)
    print("PRIOR thr:", prior_thr)
    print(prior_stats[prior_stats["split"]=="TEST"][["year","n_regime","regime_extreme_rate","lift","regime_capture_of_extremes"]].to_string(index=False))
    # 2) Rolling prior only (if you have it)
    roll_stats, roll_thr = forward_eval_onefactor(df, "abs_reaction_p75_rolling", q=0.90)
    print("ROLL thr:", roll_thr)
    print(roll_stats[roll_stats["split"]=="TEST"][["year","n_regime","regime_extreme_rate","lift","regime_capture_of_extremes"]].to_string(index=False))
    # 3) Prior + fragility (your current core)
    pf_stats, (p_thr, f_thr) = forward_eval_twofactor_and(df, "abs_reaction_p75", "momentum_fragility_score", q=0.90)
    print("PRIOR+FRAG thrs:", p_thr, f_thr)
    print(pf_stats[pf_stats["split"]=="TEST"][["year","n_regime","regime_extreme_rate","lift","regime_capture_of_extremes"]].to_string(index=False))

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
    return

if __name__ == "__main__":
    print("Running Backtesting Stage")
    df = run_pipeline()
    backtesting_suite(df)
    print("-------------------------------\nBacktesting Done.")