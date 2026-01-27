# pipeline/stage3.py
"""
    Risk Scoring and reccomendation stage
"""
import pandas as pd

from risk_scoring.scoring_features import engineer_vol_stress

def stage3(input_df):
    df = input_df.copy()
    scoring_df = pd.DataFrame()
    features = [
        engineer_vol_stress
    ]

    for f in features:
        df = f(df)

    if df is None:
        raise ValueError("\n---ERROR! Stage 3 Returned None.---\n")
    return df