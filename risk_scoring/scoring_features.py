# risk_scoring/scoring_features.py
import numpy as np
import pandas as pd


def engineer_vol_stress(input_df):
    """
        If vol_ratio_10_to_30 is high, recent vol spiked relative to the recent baseline → “stress”.
        Define “stress” as “top X%”.
        Typical starting points:
        Top 20% = “elevated”
        Top 10% = “high”
        Top 5% = “extreme”

        Adds percentile-based volatility stress flags using cross-sectional distribution per date.
        Leakage-safe if ratio_col is already computed using shift(1) rolling stats.

        Output columns:
        - vol_ratio_cs_pct: cross-sectional percentile rank on that date (0..1)
        - vol_stress_high: top (1-q_extreme)
        - vol_stress_elevated: top (1-q_high)
    """
    df = input_df.copy()
    group = "date",
    q_high = 0.80,   # elevated
    q_extreme = 0.90 # high / extreme

    replaced = df["vol_ratio_10_to_30"].replace( [np.inf, -np.inf], np.nan)
    df["vol_ratio_10_to_30"] = replaced





    return  