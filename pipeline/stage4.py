# pipeline/stage4.py
from risk_scoring.scoring_features import (
    engineer_large_reaction,
    engineer_extreme_reaction,
    engineer_vol_stress, 
    engineer_momentum_pressure,
    engineer_earnings_explosiveness,
    engineer_timing_danger,
    engineer_proximity_score,
    engineer_vol_expansion_score,
    engineer_momentum_fragility_score,
    engineer_earnings_explosiveness_score,
    engineer_timing_danger_score,
    classify_large_relative_earnings_move_bucket)
from risk_scoring.scoring_features_sector import engineer_sector_vol_stress
def stage4(stage3_df):
    """
        Risk Scoring and recommendation stage
        Returns a separate DF
    """
    print("--------------------\nStage 4 - Risk Scoring...")
    stage4_df = stage3_df.copy()
    features = [
        engineer_large_reaction,
        engineer_extreme_reaction,
        engineer_vol_stress,
        engineer_sector_vol_stress,
        engineer_momentum_pressure,
        engineer_earnings_explosiveness,
        engineer_timing_danger,
        engineer_proximity_score,
        engineer_vol_expansion_score,
        engineer_momentum_fragility_score,
        engineer_earnings_explosiveness_score,
        engineer_timing_danger_score,
        classify_large_relative_earnings_move_bucket
    ]
    for f in features:
        stage4_df = f(stage4_df)
    if stage4_df is None:
        raise ValueError("\n---ERROR! Stage 4 Returned None.---\n")
    print("Stage 4 DONE")
    return stage4_df