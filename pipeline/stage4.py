# pipeline/stage4.py
from risk_scoring.scoring_features import (engineer_large_reaction,
                                           engineer_extreme_reaction,
                                           engineer_vol_stress, 
                                           engineer_momentum_pressure,
                                           engineer_earnings_explosiveness,
                                           engineer_timing_danger)
from risk_scoring.scoring_features_sector import engineer_sector_vol_stress

def stage4(stage3_df):
    """
        Risk Scoring and recommendation stage
        Returns a separate DF
    """
    print("Stage 4...")
    df = stage3_df.copy()
    features = [
        engineer_large_reaction,
        engineer_extreme_reaction,
        engineer_vol_stress,
        engineer_sector_vol_stress,
        engineer_momentum_pressure,
        engineer_earnings_explosiveness,
        engineer_timing_danger
    ]
    for f in features:
        df = f(df)
    if df is None:
        raise ValueError("\n---ERROR! Stage 3 Returned None.---\n")
    with open("columns.txt","w") as report:
        report.write(str(list(df.columns)))
    return df