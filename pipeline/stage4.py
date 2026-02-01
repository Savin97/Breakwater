# pipeline/stage4.py
"""    
    Stage 4 - Back testing
    Produces credibility tables (calibration, lift, hit rates, bucket stats, stability by year/sector).
"""
from back_testing.back_testing import classify_large_earnings_move
from back_testing.back_testing_features import (engineer_abs_reaction_3d,
                                                engineer_rolling_abs_reaction_p75_rolling)

def stage4(stage3_df):
    df = stage3_df.copy()
    df = engineer_abs_reaction_3d(df)
    df = engineer_rolling_abs_reaction_p75_rolling(df)
    df = classify_large_earnings_move(df)

    return df