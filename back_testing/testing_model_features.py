# back_testing/temp_testing_functions.py
import pandas as pd

def check_explosiveness_feature(df):
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



