# back_testing/features_for_backtesting.py
import numpy as np

def add_joint_regime_flag(df,threshold):
    earnings = df[df["is_earnings_day"] == 1].copy()

    exp_thr = earnings["earnings_explosiveness_score"].quantile(threshold)
    frag_thr = earnings["momentum_fragility_score"].quantile(threshold)

    df = df.copy()
    df["is_joint_regime"] = (
        (df["earnings_explosiveness_score"] >= exp_thr) &
        (df["momentum_fragility_score"] >= frag_thr) &
        (df["is_earnings_day"])
    ).astype(int)

    return df