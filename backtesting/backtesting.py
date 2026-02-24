# backtesting/backtesting.py
import numpy as np

def backtesting_suite(backtesting_df):
    from backtesting.GPT_GENERATED_FILES.CHATGENERATED_stage5 import stage5_part_a, stage5_part_b
    from backtesting.GPT_GENERATED_FILES.year_by_year_regime_eval import run_regime_eval 
    from backtesting.GPT_GENERATED_FILES.year_by_year_regular import main as year_by_year_main
    #stage5_df = stage5_part_a(backtesting_df)
    #stage5_b_df = stage5_part_b(backtesting_df)
    #stage5_year_by_year_df = year_by_year_main(backtesting_df)
    stage5_year_by_year_holds_thresh = run_regime_eval(
        backtesting_df, mode="forward",
        train_years="2005-2010", test_years="2011-2025",
        report_train=True
    )
    print(stage5_year_by_year_holds_thresh[
        ["split","year","N_earnings","baseline_extreme_rate",
        "n_regime","regime_extreme_rate","lift", 
        "regime_share_of_events","regime_capture_of_extremes"]
    ].to_string(index=False))

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

    def regime_confusion_metrics(df):
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

    for threshold in [0.5, 0.7, 0.8, 0.85, 0.9, 0.95]:
        print(f"\n--- Joint Regime Flag at {threshold} Quantile ---")
        backtesting_df = add_joint_regime_flag(backtesting_df, threshold)
        regime_confusion_metrics(backtesting_df)

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
    comparing_regime_results_to_volatility_only(backtesting_df)


    