# back_testing/back_testing.py
import numpy as np

def back_testing_suite(back_testing_df):
        from back_testing.GPT_GENERATED_FILES.CHATGENERATED_stage5 import stage5_part_a, stage5_part_b
        from back_testing.GPT_GENERATED_FILES.year_by_year_regime_eval import run_regime_eval 
        from back_testing.GPT_GENERATED_FILES.year_by_year_regular import main as year_by_year_main
        #stage5_df = stage5_part_a(back_testing_df)
        #stage5_b_df = stage5_part_b(back_testing_df)
        #stage5_year_by_year_df = year_by_year_main(back_testing_df)
        stage5_year_by_year_holds_thresh = run_regime_eval(
            back_testing_df, mode="forward",
            train_years="2005-2010", test_years="2011-2025",
            report_train=True
        )
        print(stage5_year_by_year_holds_thresh[
            ["split","year","N_earnings","baseline_extreme_rate","n_regime","regime_extreme_rate","lift"]
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
            back_testing_df = add_joint_regime_flag(back_testing_df, threshold)
            regime_confusion_metrics(back_testing_df)



    