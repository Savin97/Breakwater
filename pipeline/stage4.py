# pipeline/stage4.py
"""    
    Stage 4 - Back testing
    Produces credibility tables (calibration, lift, hit rates, bucket stats, stability by year/sector).
"""
import pandas as pd

from back_testing.back_testing import back_testing_suite
from back_testing.back_testing_features import (engineer_abs_reaction_3d,
                                                engineer_abs_reaction_p75_rolling,
                                                engineer_abs_reaction_p90_rolling,
                                                classify_large_earnings_move_bucket)
from back_testing.testing_model_features import (check_explosiveness_feature, 
                                                 three_way_regime_test,
                                                 conditional_hit_rate_analysis)

def stage4(stage3_df):
    df = stage3_df.copy()
    features = [engineer_abs_reaction_3d,
                engineer_abs_reaction_p75_rolling,
                engineer_abs_reaction_p90_rolling,
                classify_large_earnings_move_bucket]
    for f in features:
        df = f(df)
        
    #back_testing_suite(df)
    ### TODO: CHECKS - TEMPORARY!
    print("\n\nCHECKS ")
    """ 
        You now have your first defensible rule:
        Tail risk increases materially when:
        timing_danger is high
        stock_vs_sector_vol ≥ 1
    """
    #test = three_way_regime_test(df)
def evaluate_high_risk_earnings_regime(df):
    """
        Define a “high-risk regime” where all three conditions hold:
        timing_danger in top 20%
        stock_vs_sector_vol >= 1
        earnings_explosiveness_score in top 10%

        Prints regime statistics and robustness checks.
    """
    subset = df[df["is_earnings_day"] == True].copy()

    cond = (
        (subset["timing_danger"] >= subset["timing_danger"].quantile(0.8)) &
        (subset["stock_vs_sector_vol"] >= 1) &
        (subset["earnings_explosiveness_score"] >= subset["earnings_explosiveness_score"].quantile(0.9))
    )

    print("Sample size:", cond.sum())
    print("Large move rate:", subset.loc[cond, "is_large_reaction"].mean())
    print("Extreme move rate:", subset.loc[cond, "is_extreme_reaction"].mean())

    print("Baseline extreme:", subset["is_extreme_reaction"].mean())
    print("Baseline large:", subset["is_large_reaction"].mean())

    print(subset.loc[cond, "abs_reaction_3d"].describe())

    print(subset.loc[cond].groupby("stock").size().describe())

    for td_q in [0.7, 0.8]:
        for ex_q in [0.8, 0.9]:
            cond = (
                (subset["timing_danger"] >= subset["timing_danger"].quantile(td_q)) &
                (subset["earnings_explosiveness_score"] >= subset["earnings_explosiveness_score"].quantile(ex_q)) &
                (subset["stock_vs_sector_vol"] >= 1)
            )

            if cond.sum() > 30:
                print(td_q, ex_q,
                    cond.sum(),
                    subset.loc[cond, "is_extreme_reaction"].mean())


    return df
