# backtesting/backtesting.py
import pandas as pd, warnings
from backtesting.testing_functions import (
    check_explosiveness_feature,
    check_feature_connection_to_large_reacion_metric,
    check_corr_of_features,
    check_score_metric,
    conditional_hit_rate_analysis,
    three_way_regime_test,
    breakwater_regime_test,
    volatility_only_regime_test,
    evaluate_high_risk_earnings_regime,
    comparing_regime_results_to_volatility_only,
    regime_confusion_metrics,
    check_feature_train_test,
    yearly_oos_report,
    forward_eval_onefactor,
    forward_eval_twofactor)

def backtesting_suite(input_df):
    """
        Backtesting Stage
        Three-way comparison:
        1. Baseline:  earnings_explosiveness_score  (historical tail profile only)
        2. New:       gated_explosiveness_score      (multiplicative vol gate on top)
        3. AND ctrl:  abs_reaction_p75 AND stock_vs_sector_vol (shows why AND regimes fail OOS)
    """
    df = input_df.copy()
    cols = ["year", "n_regime", "regime_extreme_rate", "lift", "regime_capture_of_extremes"]

    print("-------------------------------\nBASELINE: earnings_explosiveness_score")
    e_stats, _ = forward_eval_onefactor(df, "earnings_explosiveness_score", q=0.90)
    print(e_stats[e_stats["split"] == "TEST"][cols].to_string(index=False))

    print("-------------------------------\nNEW: gated_explosiveness_score")
    g_stats, _ = forward_eval_onefactor(df, "gated_explosiveness_score", q=0.90)
    print(g_stats[g_stats["split"] == "TEST"][cols].to_string(index=False))

    print("-------------------------------\nAND CTRL: abs_reaction_p75 AND stock_vs_sector_vol")
    and_stats, _ = forward_eval_twofactor(df, "abs_reaction_p75", "stock_vs_sector_vol", q=0.90)
    print(and_stats[and_stats["split"] == "TEST"][cols].to_string(index=False))

    print("-------------------------------\nyearly_oos_report (gated):")
    yearly_oos_report(df, date_col="date", score_feature="gated_explosiveness_score", target_col="abs_reaction_3d")

    return

if __name__ == "__main__":
    print("Running Backtesting Stage")
    warnings.filterwarnings('ignore')
    df = pd.read_parquet("output/full_df.parquet")
    backtesting_suite(df)
    print("-------------------------------\nBacktesting Done.")