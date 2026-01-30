# pipeline/stage3.py
"""
    Risk Scoring and recommendation stage
    Returns a separate DF
"""
import pandas as pd

from risk_scoring.scoring_features import (engineer_vol_stress, 
                                           engineer_momentum_pressure,
                                           engineer_earnings_explosiveness,
                                           engineer_timing_danger)
from risk_scoring.scoring_features_sector import engineer_sector_vol_stress

def stage3(stage2_df):
    stage3_df = stage2_df.copy()
    features = [
        engineer_vol_stress,
        engineer_sector_vol_stress,
        engineer_momentum_pressure,
        engineer_earnings_explosiveness,
        engineer_timing_danger
    ]
    
    for f in features:
        stage3_df = f(stage3_df)

    if stage3_df is None:
        raise ValueError("\n---ERROR! Stage 3 Returned None.---\n")

    return stage3_df