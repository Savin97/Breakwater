# pipeline/stage4.py
"""    
    Stage 4 - Back testing
    Produces credibility tables (calibration, lift, hit rates, bucket stats, stability by year/sector).
"""
import pandas as pd

from back_testing.back_testing import classify_large_earnings_move
from back_testing.back_testing_features import (engineer_abs_reaction_3d,
                                                engineer_rolling_abs_reaction_p75_rolling,
                                                engineer_rolling_abs_reaction_p90_rolling)

def stage4(stage3_df):
    df = stage3_df.copy()
    df = engineer_abs_reaction_3d(df)
    df = engineer_rolling_abs_reaction_p75_rolling(df)
    df = engineer_rolling_abs_reaction_p90_rolling(df)
    df = classify_large_earnings_move(df)

    bt = df.copy()
    # Earnings-only rows (use your actual column name)
    bt = bt[bt["is_earnings_day"] == 1].copy()

    # Safety: drop rows missing what you need
    bt = bt.dropna(subset=["timing_danger", "earnings_move_bucket"])

    # 10 deciles: 1..10
    bt["danger_decile"] = pd.qcut(
        bt["timing_danger"],
        q=10,
        labels=False,
        duplicates="drop" # avoids errors if timing_danger has repeated values
    ) + 1

    # Define two targets:
        # large+ = bucket >= 1
        # extreme = bucket == 2

        # Interpretation rules:
        # p_large_plus should generally increase from D1 → D10.
        # lift_large_plus in D10:
        # ~1.0 = no signal
        # 1.3–1.7 = usable
        # 2.0+ = strong
        # p_extreme is rarer, so it’ll be noisier; lift matters more than smoothness.
    bt["is_large_plus"] = (bt["earnings_move_bucket"] >= 1).astype(int)
    bt["is_extreme"]    = (bt["earnings_move_bucket"] == 2).astype(int)

    decile_table = (
        bt.groupby("danger_decile")
        .agg(
            n=("timing_danger", "size"),
            avg_danger=("timing_danger", "mean"),
            p_large_plus=("is_large_plus", "mean"),
            p_extreme=("is_extreme", "mean"),
        )
        .reset_index()
    )

    # Add lift vs overall baseline
    base_large = bt["is_large_plus"].mean()
    base_ext   = bt["is_extreme"].mean()

    decile_table["lift_large_plus"] = decile_table["p_large_plus"] / base_large
    decile_table["lift_extreme"]    = decile_table["p_extreme"] / base_ext

    decile_table.sort_values("danger_decile")

    from math import sqrt
    import numpy as np
    def wilson_ci(p, n, z=1.96):
        if n == 0:
            return (np.nan, np.nan)
        denom = 1 + z**2/n
        center = (p + z**2/(2*n)) / denom
        half = (z * sqrt((p*(1-p) + z**2/(4*n))/n)) / denom
        return (center - half, center + half)

    rows = []
    for _, r in decile_table.iterrows():
        n = int(r["n"])
        lo, hi = wilson_ci(r["p_large_plus"], n)
        rows.append((lo, hi))

    decile_table["p_large_plus_ci_lo"] = [x[0] for x in rows]
    decile_table["p_large_plus_ci_hi"] = [x[1] for x in rows]

    print(decile_table)

    print(bt["timing_danger"].describe())
    print(bt["timing_danger"].nunique())


    return df