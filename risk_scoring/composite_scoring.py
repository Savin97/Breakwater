import numpy as np

def score_proximity(df, horizon=30, power=1.5):
    """
        Pre-earnings proximity only.
        - days_to_earnings >= horizon -> 0
        - days_to_earnings <= 1 -> near 100 (but never applied on earnings day)
    """
    days_to_earnings = df["days_to_earnings"]
    
    # Only pre-earnings days (strictly > 0)
    pre_earnings_days = days_to_earnings > 0

    base = np.zeros(len(df), dtype=float)

    # normalize 1...horizon -> 1...0 (close -> high)
    x = 1 - np.clip(days_to_earnings[pre_earnings_days] / horizon, 0, 1)
    x = x ** power

    # Optional small discrete boosts (still only pre-earnings)
    boost = np.zeros(x.shape[0], dtype=float)
    near = x > 0.6
    boost += ((near & df.loc[pre_earnings_days, "is_earnings_window"]).astype(float)) * 0.05
    boost += ((near & df.loc[pre_earnings_days, "is_earnings_week"]).astype(float))   * 0.08

    base[pre_earnings_days] = np.clip(x + boost, 0, 1)
    proximity_score = 100 * base
    
    return proximity_score


    # # Normalize signals
    # # days_to_earnings >= 30  → 0   # days_to_earnings <= 0   → 100
    # base = 1 - np.clip(df["days_to_earnings"] / 30 , 0, 1)

    # # Non-linear pressure near earnings
    # base = base ** 1.5

    # boost = np.zeros(len(df))
    # near = base > 0.6
    # boost += ((near & df["is_earnings_window"]).astype(float)) * 0.05
    # boost += ((near & df["is_earnings_week"]).astype(float))   * 0.08
    # boost += ((near & df["is_earnings_day"]).astype(float))    * 0.15


    # proximity_score = 100 * np.clip(base + boost, 0, 1)
    # return proximity_score

def score_vol_expansion(df):
    """
        Vol Expansion Risk = “Is volatility already stretched before earnings?”        
        Range: 0-100
        Meaning:
        0-30 → calm
        30-60 → warming up
        60-80 → unstable
        80-100 → volatility already breaking
    """
    # Normalize signals
    z1 = df["vol_ratio_cross_sectional_pct"].fillna(0).clip(0, 1)
    z2 = ((df["stock_vs_sector_vol"].fillna(1) - 1) / 1.5).clip(0, 1)
    z3 = df["sector_vol_ratio_pct"].fillna(0).clip(0, 1)

    base = 0.40 * z1 + 0.35 * z2 + 0.25 * z3  # 0..1-ish

    elevated = df["vol_stress_elevated"].fillna(0).astype(float)
    extreme  = df["vol_stress_extreme"].fillna(0).astype(float)
    sector_h = df["sector_vol_stress_high"].fillna(0).astype(float)

    # multiplier (soft) + small additive sector bump
    multiplier = 1 + 0.25 * elevated + 0.50 * extreme
    additive  = 0.08 * sector_h

    vol_expansion = 100 * np.clip(base * multiplier + additive, 0, 1)
    return vol_expansion

    # # snap the score upward when conditions are structurally dangerous
    # boost = 0.0
    # boost = (
    #     df["vol_stress_elevated"].fillna(0).astype(float) * 0.20 +
    #     df["vol_stress_extreme"].fillna(0).astype(float)  * 0.40 +
    #     df["sector_vol_stress_high"].fillna(0).astype(float) * 0.20
    # )

    # base = 0.4 * z1 + 0.35 * z2 + 0.25 * z3
    # # score = 100*clip(base*(1+0.5*extreme+0.25*elevated)+0.1*sector_high, 0, 1)
    # vol_expansion = 100 * np.clip(base + boost, 0, 1)
    # return vol_expansion

def score_earnings_explosiveness(df):
    """ 
        When this stock moves on earnings, how violent can it get?
        earnings_explosiveness_score identifies stocks prone to large earnings reactions.
    """

    # Normalize signals
    #/ 3 ≈ “3σ is extreme”
    e1 = (df["earnings_explosiveness_z"].fillna(0) / 3).clip(0, 1)
    e2 = (df["earnings_tail_z"].fillna(0) / 3).clip(0, 1)
    e3 = (df["abs_reaction_p75"].fillna(0) / 0.12).clip(0, 1)  # 12% is already huge
    e4 = np.clip(df["reaction_entropy"], 0, 1) # entropy already assumed 0–1

    base = (
        0.35 * e2 +   # tail risk
        0.30 * e1 +   # overall explosiveness
        0.25 * e3 +   # large typical moves
        0.10 * e4     # chaos / unpredictability
    )

    earnings_explosiveness_score = 100 * np.clip(base, 0, 1)
    return earnings_explosiveness_score

def score_momentum_fragility(df):
    """
        Is price positioning fragile right now?
        High score = price is balanced on a knife-edge.
    """
    # Mapping momentum fragility strings to floats
    PRESSURE_MAP = {
        "normal": 0.0,
        "short_term_extreme": 0.6,
        "trend_extreme": 0.75,
        "crowded_trend": 1.0,
    }
    #Interpretation: 
    # pressure → crowding / exhaustion
    # bias → one-sided positioning
    # sector drift → late-cycle momentum

    bias_scale = df["directional_bias"].abs().quantile(0.90)


    m1 = df["momentum_pressure_regime"].map(PRESSURE_MAP).fillna(0)
    m2 = np.clip(np.abs(df["directional_bias"].fillna(0)) / bias_scale, 0, 1) # this achieves: 90% of observations live in [0,1),Top 10% saturate at 1, No arbitrary magic number, Stable across stocks and time

    m3 = np.clip(np.abs(df["sector_drift_60d"].fillna(0)) / 0.10, 0, 1)

    #m2 = (df["directional_bias"].abs() / bias_scale).clip(0, 1)
    base = (
        0.45 * m1 +   # positioning pressure
        0.35 * m3 +   # directional skew
        0.20 * m2    # sector trend maturity
    )

    momentum_fragility_score = 100 * np.clip(base, 0, 1)
    return momentum_fragility_score