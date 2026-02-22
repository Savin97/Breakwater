# pipeline/backtesting_stage.py
"""    
    Backtesting Stage
    Produces credibility tables (calibration, lift, hit rates, bucket stats, stability by year/sector).
"""
from backtesting.backtesting import backtesting_suite
from backtesting.backtesting_features import (engineer_abs_reaction_3d,
                                              engineer_abs_reaction_p75_rolling,
                                              engineer_abs_reaction_p90_rolling,
                                              classify_large_earnings_move_bucket)
from backtesting.testing_functions import (check_explosiveness_feature, 
                                           three_way_regime_test,
                                           conditional_hit_rate_analysis,
                                           volatility_only_regime_test,
                                           breakwater_regime_test)

def backtesting_stage(stage3_df):
    df = stage3_df.copy()
    features = [engineer_abs_reaction_3d,
                engineer_abs_reaction_p75_rolling,
                engineer_abs_reaction_p90_rolling,
                classify_large_earnings_move_bucket]
    for f in features:
        df = f(df)

    print("Volatility Only:\n")
    print(volatility_only_regime_test(df))
    print("\nBreakwater Regime:\n")
    print(breakwater_regime_test(df))
    print("\nBreakwater Regime:\n")    
    backtesting_suite(df)
    print("\nThree way regime test:\n")
    print(three_way_regime_test(df))
    """ 
        You now have your first defensible rule:
        Tail risk increases materially when:
        timing_danger is high
        stock_vs_sector_vol ≥ 1
    """
    
