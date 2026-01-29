# pipeline/stage3.py
"""
    Risk Scoring and recommendation stage
    Returns a separate DF
"""
import pandas as pd

from risk_scoring.scoring_features import (engineer_vol_stress, 
                                           engineer_momentum_pressure,
                                           engineer_earnings_explosiveness)
from risk_scoring.scoring_features_sector import engineer_sector_vol_stress

def stage3(stage2_df):
    stage3_df = stage2_df.copy()
    features = [
        engineer_vol_stress,
        engineer_sector_vol_stress,
        engineer_momentum_pressure ,
        engineer_earnings_explosiveness       
    ]
    
    for f in features:
        stage3_df = f(stage3_df)

    if stage3_df is None:
        raise ValueError("\n---ERROR! Stage 3 Returned None.---\n")
    
    subset = ["stock", "date", "price", "earnings_date", "sector", "sub_sector",
            "is_earnings_day","vol_ratio_cross_percentile",
            "vol_stress_elevated", "vol_stress_extreme", "sector_vol_ratio_pct",
            "sector_vol_stress_high"]


    return stage3_df