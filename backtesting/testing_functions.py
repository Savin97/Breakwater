# back_testing/temp_testing_functions.py
import pandas as pd

def check_explosiveness_feature(df):
    """ 
        Extreme move rate: 0.10421515386604603
        explosive_bucket
        (18.035, 46.526]    0.066775
        (46.526, 57.006]    0.035889
        (57.006, 68.114]    0.037520
        (68.114, 80.606]    0.089723
        (80.606, 100.0]     0.307818

        The highest explosiveness bucket gives:
        P(extreme move | top explosiveness) ≈ 31%
        vs baseline:
        P(extreme move) ≈ 10%

        That's roughly a 3x risk multiplier.
        That's strong signal for a single feature. Very strong, actually.

        This gives results:
        “Stocks with historically explosive earnings behavior are about 
        three times more likely to experience an extreme earnings move.”

        Why this validates your explosiveness design
        It confirms:

        median reaction component is useful
        p75 tail component is useful
        normalization by vol_30d works
        scoring compression didn't kill signal
        requiring 8 historical earnings events was reasonable

        All of that survived testing.

        You just discovered something real about earnings risk:
        Extreme earnings moves are heavily stock-dependent.
        Some stocks are simply “earnings explosives.”

    """
    # Check correlation structure of timing_danger and its components
    subset = df[df["is_earnings_day"] == True].copy()

    subset["is_extreme_reaction"] = (
        subset["abs_reaction_3d"] >= 0.08
    ).astype(int)

    print("Extreme move rate:", subset["is_extreme_reaction"].mean())

    subset["explosive_bucket"] = pd.qcut(
        subset["earnings_explosiveness_score"],
        5,
        duplicates="drop"
    )

    print(
        subset.groupby("explosive_bucket")["is_extreme_reaction"].mean()
    )
    print(subset.groupby("explosive_bucket").size())

    top = subset[subset["earnings_explosiveness_score"] >= subset["earnings_explosiveness_score"].quantile(0.8)]
    bottom = subset[subset["earnings_explosiveness_score"] <= subset["earnings_explosiveness_score"].quantile(0.2)]

    print("Top bucket extreme rate:", top["is_extreme_reaction"].mean())
    print("Bottom bucket extreme rate:", bottom["is_extreme_reaction"].mean())

def three_way_regime_test(df):
    """
        This is a temporary function to test the three-way regime hypothesis:
        Tail risk increases materially when:
        timing_danger is high
        stock_vs_sector_vol ≥ 1
        earnings_explosiveness_score is in the top decile

        We will check the hit rates for large_plus and extreme earnings moves across different regimes.
    """
    # separate back testing df to only earnings days
    bt = df.copy()
    bt = bt[bt["is_earnings_day"] == 1].copy()

    bt["is_large_plus"] = (bt["earnings_move_bucket"] >= 1).astype(int)
    bt["is_extreme"]    = (bt["earnings_move_bucket"] == 2).astype(int)

    # Define the three regime conditions:

    # High timing danger (top decile)
    danger_cutoff = bt["timing_danger"].quantile(0.9)
    bt["high_danger"] = (bt["timing_danger"] >= danger_cutoff).astype(int)

    # High individual volatility
    bt["high_individual_vol"] = (bt["stock_vs_sector_vol"] >= 1).astype(int)

    # High explosiveness (top decile)
    exp_cut = bt["earnings_explosiveness_score"].quantile(0.9)
    bt["high_explosiveness"] = (
        bt["earnings_explosiveness_score"] >= exp_cut
    ).astype(int)

    baseline = {
        "group": "ALL",
        "n": len(bt),
        "p_large_plus": bt["is_large_plus"].mean(),
        "p_extreme": bt["is_extreme"].mean(),
    }

    mask = (
        (bt["high_danger"] == 1) &
        (bt["high_individual_vol"] == 1) &
        (bt["high_explosiveness"] == 1)
        )

    three_way = {
        "group": "High danger + individual vol + explosiveness",
        "n": mask.sum(),
        "p_large_plus": bt.loc[mask, "is_large_plus"].mean(),
        "p_extreme": bt.loc[mask, "is_extreme"].mean(),
    }

    return pd.DataFrame([baseline, three_way])


def conditional_hit_rate_analysis(df):
    """ 
        We already proved something important:
        Unconditional timing_danger ≠ large move predictor
        So now we ask the correct question:
        In which regimes does earnings risk actually turn into realized large moves?
        That means conditioning.
    """
    bt = df.copy()
    bt = bt[bt["is_earnings_day"] == 1].copy()

    # We want:
    # vol_stress_extreme == 1
    # Sector earnings crowding: sector_earnings_density >= median
    # Stock-specific volatility dominance: stock_vs_sector_vol >= 1
    bt["is_large_plus"] = (bt["earnings_move_bucket"] >= 1).astype(int)
    bt["is_extreme"]    = (bt["earnings_move_bucket"] == 2).astype(int)

    # median split for density
    density_med = bt["sector_earnings_density"].median()
    bt["dense_earnings"] = (bt["sector_earnings_density"] >= density_med).astype(int)

    bt["high_individual_vol"] = (bt["stock_vs_sector_vol"] >= 1).astype(int)

    danger_cut = bt["timing_danger"].quantile(0.9)
    bt["high_danger"] = (bt["timing_danger"] >= danger_cut).astype(int)

    def cond_table(df, mask, label):
        sub = df[mask]
        return {
            "group": label,
            "n": len(sub),
            "p_large_plus": sub["is_large_plus"].mean(),
            "p_extreme": sub["is_extreme"].mean(),
        }

    rows = []

    # baseline
    rows.append(cond_table(bt, bt.index == bt.index, "ALL"))

    # danger only
    rows.append(cond_table(bt, bt["high_danger"] == 1, "High danger"))

    # danger + vol stress
    rows.append(cond_table(
        bt,
        (bt["high_danger"] == 1) & (bt["vol_stress_extreme"] == 1),
        "High danger + vol stress extreme"
    ))

    # danger + low density
    rows.append(cond_table(
        bt,
        (bt["high_danger"] == 1) & (bt["dense_earnings"] == 0),
        "High danger + low earnings density"
    ))

    # danger + idio vol
    rows.append(cond_table(
        bt,
        (bt["high_danger"] == 1) & (bt["high_individual_vol"] == 1),
        "High danger + high idio vol"
    ))

    temp_df = pd.DataFrame(rows)

    temp_df.to_csv("temp_df.csv",index=False)

def check_timing_danger_connection_to_earnings_move_bucket(df):
    
    ## Diagnostic: timing_danger connection to earnings_move_bucket
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
        # p_large_plus should generally increase from D1 -> D10.
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

    print(bt)
    print("----------------------")
    print(decile_table)
    print("----------------------")

    print(bt["timing_danger"].describe())
    print(bt["timing_danger"].nunique())


def check_corr_of_features(df):
    # only earnings ??? 
    df_to_check = df[df["is_earnings_day"] == 1].copy
    print(df_to_check[
        ["proximity_score",
        "vol_expansion_score",
        "momentum_fragility_score",
        "earnings_explosiveness_score",
        "timing_danger"]
    ].corr())

def breakwater_regime_test(df):
    bt = df.copy()
    bt = bt[bt["is_earnings_day"] == 1].copy()

    bt["is_large_plus"] = (bt["earnings_move_bucket"] >= 1).astype(int)
    bt["is_extreme"]    = (bt["earnings_move_bucket"] == 2).astype(int)

    # timing danger top decile
    danger_cutoff = bt["timing_danger"].quantile(0.9)
    bt["high_danger"] = (bt["timing_danger"] >= danger_cutoff)

    # explosiveness top decile
    explosiveness_cutoff = bt["earnings_explosiveness_score"].quantile(0.9)
    bt["high_explosive"] = (bt["earnings_explosiveness_score"] >= explosiveness_cutoff)

    bt["vol_vs_sector"] = (bt["stock_vs_sector_vol"] >= 1)

    regime = (bt["vol_vs_sector"]
    )

    subset = bt[regime]

    results = {
        "n_events": len(subset),
        "extreme_rate": subset["is_extreme"].mean(),
        "large_plus_rate": subset["is_large_plus"].mean(),
    }

    return results

def volatility_only_regime_test(df):
    """
        Test whether volatility alone can generate high extreme-move rates.
    """

    bt = df.copy()
    bt = bt[bt["is_earnings_day"] == 1].copy()

    # Labels
    bt["is_large_plus"] = (bt["earnings_move_bucket"] >= 1).astype(int)
    bt["is_extreme"]    = (bt["earnings_move_bucket"] == 2).astype(int)

    # --- Define volatility regime ---

    # Top decile volatility
    vol_cutoff = bt["vol_30d"].quantile(0.9)
    bt["high_vol"] = (bt["vol_30d"] >= vol_cutoff)

    # Stock moving more than sector
    bt["vol_vs_sector"] = (bt["stock_vs_sector_vol"] >= 1)

    regime = bt["high_vol"] & bt["vol_vs_sector"]

    subset = bt[regime]

    results = {
        "n_events": len(subset),
        "extreme_rate": subset["is_extreme"].mean(),
        "large_plus_rate": subset["is_large_plus"].mean(),
    }

    return results

def evaluate_high_risk_earnings_regime(df):
    """
        Define a “high-risk regime” where all three conditions hold:
        timing_danger in top 20%
        stock_vs_sector_vol >= 1
        earnings_explosiveness_score in top 10%

        Prints regime statistics and robustness checks.
    """
    subset = df[df["is_earnings_day"] == True].copy()

    cond = (
        (subset["timing_danger"] >= subset["timing_danger"].quantile(0.8)) &
        (subset["stock_vs_sector_vol"] >= 1) &
        (subset["earnings_explosiveness_score"] >= subset["earnings_explosiveness_score"].quantile(0.9))
    )

    print("Sample size:", cond.sum())
    print("Large move rate:", subset.loc[cond, "is_large_reaction"].mean())
    print("Extreme move rate:", subset.loc[cond, "is_extreme_reaction"].mean())

    print("Baseline extreme:", subset["is_extreme_reaction"].mean())
    print("Baseline large:", subset["is_large_reaction"].mean())

    print(subset.loc[cond, "abs_reaction_3d"].describe())

    print(subset.loc[cond].groupby("stock").size().describe())

    for td_q in [0.7, 0.8]:
        for ex_q in [0.8, 0.9]:
            cond = (
                (subset["timing_danger"] >= subset["timing_danger"].quantile(td_q)) &
                (subset["earnings_explosiveness_score"] >= subset["earnings_explosiveness_score"].quantile(ex_q)) &
                (subset["stock_vs_sector_vol"] >= 1)
            )

            if cond.sum() > 30:
                print(td_q, ex_q,
                    cond.sum(),
                    subset.loc[cond, "is_extreme_reaction"].mean())

    return df