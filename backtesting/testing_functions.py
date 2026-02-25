# backtesting/testing_functions.py
import pandas as pd
import numpy as np
import matplotlib

from backtesting.features_for_backtesting import add_joint_regime_flag

"""
    Backtesting & Regime Evaluation Utilities for Breakwater

    This module contains diagnostic, validation, and regime-testing functions
    used to evaluate Breakwater's earnings-period risk framework.

    Purpose
    -------
    Provide tools to:
    • Validate individual feature predictive power
    • Test multi-factor “high-risk” regimes
    • Compare regime performance vs volatility-only baselines
    • Measure precision/recall of extreme-move detection
    • Run train/test and walk-forward robustness checks
    • Inspect distributional stability and lift statistics

    ---------------------------------------------------------------------------
    FEATURE DIAGNOSTICS
    ---------------------------------------------------------------------------

    check_explosiveness_feature(df)
        Tests earnings_explosiveness_score via quantile bucketing and compares
        large/extreme move rates across buckets.

    check_timing_danger_connection_to_earnings_move_bucket(df)
        Decile analysis of timing_danger vs large/extreme move outcomes,
        including lift vs baseline and confidence intervals.

    check_corr_of_features(df)
        Prints correlation matrix between core components:
        proximity, volatility expansion, momentum fragility,
        explosiveness, and timing_danger.

    check_timing_danger_score_metric(df)
        Evaluates distribution, dispersion, and bucketed reaction behavior
        of the timing_danger score (including normalization diagnostics).

    ---------------------------------------------------------------------------
    REGIME TESTING
    ---------------------------------------------------------------------------

    three_way_regime_test(df)
        Tests the 3-factor regime hypothesis:
        high timing_danger + elevated idiosyncratic vol +
        high explosiveness, and compares hit rates vs baseline.

    breakwater_regime_test(df)
        Computes regime statistics (event count, large/extreme rates)
        for the defined Breakwater regime mask.

    volatility_only_regime_test(df)
        Baseline comparison using volatility-only conditions
        (high vol_30d + elevated idiosyncratic vol).

    evaluate_high_risk_earnings_regime(df)
        Defines a stricter “high-risk” regime and runs threshold grids
        to evaluate sensitivity of extreme-move concentration.

    comparing_regime_results_to_volatility_only(df)
        Stress-tests regime performance vs volatility-only signals,
        including shuffled-label checks and stability diagnostics.

    ---------------------------------------------------------------------------
    CLASSIFICATION METRICS
    ---------------------------------------------------------------------------

    regime_confusion_metrics(df)
        Computes TP/FN/FP/TN, precision, recall for regime flags
        predicting extreme reactions.

    (Threshold sweep block)
        Iterates multiple quantile thresholds and evaluates confusion
        metrics for each regime definition.

    ---------------------------------------------------------------------------
    ROBUSTNESS & OUT-OF-SAMPLE VALIDATION
    ---------------------------------------------------------------------------

    check_timing_danger_train_test(df)
        Pre/post split validation ensuring normalization and bucket
        edges are trained only on prior data.

    yearly_oos_report(df, ...)
        Walk-forward yearly OOS test computing correlation,
        bucket separation, and lift statistics per year.

    All evaluations are performed on earnings-day events unless otherwise specified.
"""

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
        # p_extreme is rarer, so it'll be noisier; lift matters more than smoothness.
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
    df_to_check = df[df["is_earnings_day"] == 1].copy()
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


def regime_confusion_metrics(input_df):
    df = input_df.copy()
    for threshold in [0.5, 0.7, 0.8, 0.85, 0.9, 0.95]:
        print(f"\n--- Joint Regime Flag at {threshold} Quantile ---")
        df = add_joint_regime_flag(df, threshold)
        earnings = df[df["is_earnings_day"] == 1].copy()

        extreme = earnings["is_extreme_reaction"] == 1
        regime = earnings["is_joint_regime"] == 1

        TP = ((regime) & (extreme)).sum()
        FN = ((~regime) & (extreme)).sum()
        FP = ((regime) & (~extreme)).sum()
        TN = ((~regime) & (~extreme)).sum()

        recall = TP / (TP + FN)
        precision = TP / (TP + FP)

        print("TP:", TP)
        print("FN:", FN)
        print("FP:", FP)
        print("TN:", TN)
        print("recall:", recall)
        print("precision:", precision)
        print(regime.sum(), "events flagged")
        print(len(earnings), "total earnings events")

def comparing_regime_results_to_volatility_only(df):
    earn = df[df.is_earnings_day == 1].sort_values(["stock","date"]).copy()

    # For each stock, compare current p75 with previous event p75
    earn["p75_diff"] = (
        earn.groupby("stock")["abs_reaction_p75"]
        .diff()
    )
    print(earn[["stock","date","abs_reaction_3d","abs_reaction_p75","p75_diff"]].head(20))

    # Test 2 
    earn["extreme_shuffled"] = np.random.permutation(earn["is_extreme_reaction"].values)
    threshold = 0.9
    exp_thr = earn["earnings_explosiveness_score"].quantile(threshold)
    frag_thr = earn["momentum_fragility_score"].quantile(threshold)

    earn["is_joint_regime"] = (
        (df["earnings_explosiveness_score"] >= exp_thr) &
        (df["momentum_fragility_score"] >= frag_thr) &
        (df["is_earnings_day"])
    ).astype(int)

    regime = earn["is_joint_regime"] == 1

    TP = ((regime) & (earn["extreme_shuffled"] == 1)).sum()
    FP = ((regime) & (earn["extreme_shuffled"] == 0)).sum()

    print("Shuffled precision:", TP / (TP + FP))

    # Test 3
    median_year = earn["date"].dt.year.median()

    first_half = earn[earn["date"].dt.year <= median_year]
    second_half = earn[earn["date"].dt.year > median_year]

    def regime_precision(sub):
        regime = sub["is_joint_regime"] == 1
        extreme = sub["is_extreme_reaction"] == 1
        TP = ((regime) & (extreme)).sum()
        FP = ((regime) & (~extreme)).sum()
        return TP / (TP + FP)

    print("First half precision:", regime_precision(first_half))
    print("Second half precision:", regime_precision(second_half))

    # Top volatility test
    threshold = 0.98

    vol_thr = earn["vol_30d"].quantile(threshold)
    regime_vol_top2 = earn["vol_30d"] >= vol_thr
    print(regime_vol_top2)

    regime_vol_top2 = earn["vol_30d"] >= vol_thr
    extreme = earn["is_extreme_reaction"] == 1

    TP = ((regime_vol_top2) & (extreme)).sum()
    FP = ((regime_vol_top2) & (~extreme)).sum()

    precision = TP / (TP + FP)

    print("Top 2% vol events:", regime_vol_top2.sum())
    print("Extreme rate (precision):", precision)

def check_timing_danger_score_metric(input_df):
    df = input_df.copy()
    df["abs_reaction_3d"] = df["reaction_3d"].abs()
    df["timing_danger_score"] = (
            100 * (df["timing_danger"] - df["timing_danger"].min()) /
            (df["timing_danger"].max() - df["timing_danger"].min())
        )
    test_score_df = df[["stock", "date","earnings_date","abs_reaction_3d", "timing_danger_score"]].dropna()
    s = test_score_df["timing_danger_score"]
    stats = {
        "count": s.count(),
        "min": s.min(),
        "max": s.max(),
        "mean": s.mean(),
        "median": s.median(),
        "std": s.std(),
        "p10": s.quantile(0.10),
        "p25": s.quantile(0.25),
        "p75": s.quantile(0.75),
        "p90": s.quantile(0.90),
        "p95": s.quantile(0.95),
    }
    print(stats)
    counts, bins = np.histogram(s, bins=30)

    for i in range(len(counts)):
        print(f"{bins[i]:.4f} to {bins[i+1]:.4f} : {counts[i]}")
    per_day_dispersion = (
    test_score_df
        .groupby("earnings_date")["timing_danger_score"]
        .agg(["mean", "std", "min", "max", "count"])
    )
    per_stock = (
        test_score_df
        .groupby("stock")["timing_danger_score"]
        .agg(["mean", "std", "count"])
    )
    top_10 = s.quantile(0.90)
    bottom_10 = s.quantile(0.10)

    high_tail_rate = (s >= top_10).mean()
    print(high_tail_rate)
    low_tail_rate = (s <= bottom_10).mean()
    print(low_tail_rate)
    corr = test_score_df[["timing_danger_score", "abs_reaction_3d"]].corr()
    print(corr)

    test_score_df["bucket"] = pd.qcut(
        test_score_df["timing_danger_score"], 
        q=5, 
        labels=False
    )

    bucket_stats = test_score_df.groupby("bucket")["abs_reaction_3d"].mean()
    print(bucket_stats)

def check_timing_danger_train_test(df):
    test_score_df = df[["stock", "date","earnings_date","abs_reaction_3d", "timing_danger"]].dropna().copy()
    pre_2015 = test_score_df[test_score_df["date"] < "2015-01-01"].copy()
    post_2015 = test_score_df[test_score_df["date"] >= "2015-01-01"].copy()

    def normalize_with_train(s, train_min, train_max):
        denom = train_max - train_min
        if denom == 0:
            return pd.Series(50, index=s.index)
        return 100 * (s - train_min) / denom
    train_min = pre_2015["timing_danger"].min()
    train_max = pre_2015["timing_danger"].max()
    pre_2015["score_oos"] = normalize_with_train(
        pre_2015["timing_danger"], train_min, train_max
    )

    post_2015["score_oos"] = normalize_with_train(
        post_2015["timing_danger"], train_min, train_max
    ).clip(0, 100)
    pre_corr = pre_2015[["score_oos", "abs_reaction_3d"]].corr().iloc[0,1]
    print("Train corr:", pre_corr)

    pre_2015["bucket"] = pd.qcut(pre_2015["score_oos"], q=5, labels=False)
    print(pre_2015.groupby("bucket")["abs_reaction_3d"].mean())
    train_edges = pd.qcut(
        pre_2015["score_oos"],
        q=5,
        retbins=True,
        labels=False
    )[1]

    post_2015["bucket"] = pd.cut(
        post_2015["score_oos"],
        bins=train_edges,
        labels=False,
        include_lowest=True
    )
    post_corr = post_2015[["score_oos", "abs_reaction_3d"]].corr().iloc[0,1]
    print("Test corr:", post_corr)

    print(post_2015.groupby("bucket")["abs_reaction_3d"].mean())

def yearly_oos_report(df, date_col="date", score_col="timing_danger", target_col="abs_reaction_3d", q=5):
    d = df[[date_col, score_col, target_col]].dropna().copy()
    d[date_col] = pd.to_datetime(d[date_col])

    years = sorted(d[date_col].dt.year.unique())
    rows = []

    for y in years[1:]:  # start from 2nd year (need prior data to train)
        train = d[d[date_col] < pd.Timestamp(f"{y}-01-01")]
        test  = d[(d[date_col] >= pd.Timestamp(f"{y}-01-01")) & (d[date_col] < pd.Timestamp(f"{y+1}-01-01"))]

        if len(train) < 500 or len(test) < 100:
            continue

        train_min = train[score_col].min()
        train_max = train[score_col].max()
        denom = train_max - train_min
        if denom == 0:
            continue

        # Normalize using train params only
        train_score = 100 * (train[score_col] - train_min) / denom
        test_score  = 100 * (test[score_col]  - train_min) / denom

        # Clip to avoid out-of-range due to new extremes
        train_score = train_score.clip(0, 100)
        test_score  = test_score.clip(0, 100)

        # Freeze bucket edges from train
        _, edges = pd.qcut(train_score, q=q, retbins=True, duplicates="drop")

        test_bucket = pd.cut(test_score, bins=edges, labels=False, include_lowest=True)

        test_eval = pd.DataFrame({
            "score": test_score,
            "bucket": test_bucket,
            "target": test[target_col].values
        }).dropna(subset=["bucket"])

        if len(test_eval) < 100:
            continue

        corr = test_eval[["score", "target"]].corr().iloc[0, 1]

        bucket_means = test_eval.groupby("bucket")["target"].mean()
        # separation metrics
        top = bucket_means.max()
        bot = bucket_means.min()
        lift_abs = top - bot
        lift_ratio = (top / bot) if bot != 0 else np.nan

        rows.append({
            "year": y,
            "n_test": len(test_eval),
            "corr": corr,
            "bot_bucket_mean": bot,
            "top_bucket_mean": top,
            "lift_abs": lift_abs,
            "lift_ratio": lift_ratio,
        })

    yearly = pd.DataFrame(rows).sort_values("year")
    print(yearly)
    print("Avg corr:", yearly["corr"].mean())
    print("Median corr:", yearly["corr"].median())
    print("Pct years corr>0.15:", (yearly["corr"] > 0.15).mean())
