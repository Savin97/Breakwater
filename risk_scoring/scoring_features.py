# risk_scoring/scoring_features.py
import numpy as np
import pandas as pd

from risk_scoring.composite_scoring import (score_proximity,
                               score_vol_expansion,
                               score_earnings_explosiveness,
                               score_momentum_fragility)

def engineer_vol_stress(input_df, ratio_col: str = "vol_ratio_10_to_30"):

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
        - vol_ratio_cross_pct: cross-sectional percentile rank on that date (0..1)
        - vol_stress_high: top (1-q_extreme)
        - vol_stress_elevated: top (1-q_high)
    """
    df = input_df.copy()
    q_high = 0.80  # elevated
    q_extreme = 0.90 # high / extreme

    # Prevent div by zero
    replaced = df[ratio_col].replace( [np.inf, -np.inf], np.nan)
    df[ratio_col] = replaced

    # Cross-sectional percentile rank per day
    df["vol_ratio_cross_percentile"] = (
        df.groupby("date")[ratio_col]
            .rank(pct=True, method="average")
    )

    df["vol_stress_elevated"] = ( df["vol_ratio_cross_percentile"] >= q_high ).astype(int)
    # Might be a better implementaion 
    # (guards against only showing vol_stress_elevated as relative, not absolute):
    # vol_stress_elevated =
        # (vol_ratio_cross_percentile >= 0.80) &
        # (vol_ratio_10_to_30 >= 1.10)

    df["vol_stress_extreme"] = ( df["vol_ratio_cross_percentile"] >= q_extreme ).astype(int)
    
    return df


def engineer_momentum_pressure(input_df, quantile = 0.8) -> pd.DataFrame:
    """ 
        top 20% of absolute momentum values per date
        cross-sectional and date-aligned
        Momentum pressure measures how unusually stretched a stock's price action 
        is relative to peers ahead of earnings,
        capturing both short-term crowding and longer-term trend extension.

    Returns
    -------
        - 'momentum_pressure_regime' : str
          {'normal', 'short_term_extreme', 'trend_extreme', 'crowded_trend'}
    """
    df = input_df.copy()

    abs5 = df["mom_5d"].abs()
    abs20 = df["mom_20d"].abs()

    threshold_5 = abs5.groupby(df["date"]).transform(lambda x: x.quantile(quantile))
    threshold_20 = abs20.groupby(df["date"]).transform(lambda x: x.quantile(quantile))

    mom_5 = abs5 > threshold_5
    mom_20 = abs20 > threshold_20

    df["momentum_pressure_regime"] = np.select(
        [~mom_5 & ~mom_20,  mom_5 & ~mom_20,  ~mom_5 & mom_20,  mom_5 & mom_20],
        ["normal", "short_term_extreme", "trend_extreme", "crowded_trend"],
        default="normal"
    )
    return df


def engineer_earnings_explosiveness(input_df, epsilon = 1e-6):
    """
        Adds:
        ####- earnings_explosiveness (raw)          = abs_reaction_median_3d
        - earnings_explosiveness_z (normalized) = abs_reaction_median_3d / max(vol_30d, epsilon)
        Median-based explosiveness = “typical risk”
        
        - earnings_tail_z (optional)            = abs_reaction_p75_3d / max(vol_30d, epsilon)  (if column exists)
        P75-based explosiveness = “tail danger” (“When earnings go bad, this is how ugly it can get.”)
        
        Assumptions:
        - med_col / p75_col are already computed using ONLY past earnings events (shifted).
        - vol_30d is rolling vol from daily returns (ideally shifted by 1 day).
    """
    
    df = input_df.sort_values(["stock", "date"]).copy()

    df["earnings_explosiveness_z"] = (
        df["abs_reaction_median"] / np.maximum(df["vol_30d"], epsilon)
        )

    df["earnings_tail_z"] = (
        df["abs_reaction_p75"] / np.maximum(df["vol_30d"], epsilon)
        )

    return df

def engineer_timing_danger(input_df, w_prox=0.25, w_vol=0.25, w_mom=0.20, w_exp=0.30,):
    """
        TimingDanger = w1 * ProximityRisk +
                        + w2 * VolExpansionRisk
                        + w3 * MomentumFragility
                        + w4 * EarningsExplosiveness
    """
    df = input_df.copy()

    # TODO: better way of applying weights? makes sure they sum to 1.
    # weights = np.array([w_prox, w_vol, w_mom, w_exp], dtype=float)
    # weights = weights / weights.sum()

    # timing_danger = (
    #     weights[0] * prox +
    #     weights[1] * vol +
    #     weights[2] * mom +
    #     weights[3] * exp
    # )

    prox = score_proximity(df)
    vol  = score_vol_expansion(df)
    mom  = score_momentum_fragility(df)
    exp  = score_earnings_explosiveness(df)

    timing_danger = (
        w_prox * prox +
        w_vol  * vol  +
        w_mom  * mom  +
        w_exp  * exp
    )

    scores = {
        "proximity_score": prox,
        "vol_expansion_score": vol,
        "momentum_fragility_score": mom,
        "earnings_explosiveness_score": exp,
        "timing_danger": np.clip(timing_danger, 0, 100),
    }

    df = df.assign(**scores)

    return df
    
    
# def timing_danger_bucket(td):
#     # td: Series or scalar in [0, 100]

#     # df["timing_danger_bucket"] = pd.cut(
#     #     df["timing_danger"],
#     #     bins=[0, 30, 55, 75, 100],
#     #     labels=["Low", "Moderate", "High", "Extreme"]
#     # )
#     return pd.cut(
#         td,
#         bins=[-1, 20, 40, 60, 80, 101],
#         labels=["Very Low", "Low", "Moderate", "High", "Extreme"]
#     )