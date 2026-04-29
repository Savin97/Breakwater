import pandas as pd, numpy as np
import warnings
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
# stage3_df = pd.read_parquet("stage3_df.parquet")
full_df = pd.read_parquet("output/full_df.parquet")

def testing_scores(df):
    print("Running Score Testing...\n--------------------")

    stock_list = pd.read_csv("data/stock_list.csv")
    first_30_stocks = stock_list.iloc[1:31,0]

    global_earnings_df = df[df["is_earnings_day"] == 1].copy()
    
    P_extreme_global  = global_earnings_df["is_extreme_reaction"].mean()
    P_extreme_given_bucket = global_earnings_df.groupby("earnings_explosiveness_bucket")["is_extreme_reaction"].mean()
    bucket_stats = pd.DataFrame({
        "global_hist_prob": P_extreme_given_bucket,
        "global_risk_lift_vs_baseline": P_extreme_given_bucket / P_extreme_global
    })
    
    report_txt = open("report_txt.txt", "w")
    for stock in first_30_stocks:
        print(f"{stock}")
        stock_df = df[df["stock"] == stock]
        earnings_df = stock_df[stock_df["is_earnings_day"] == 1]
        latest_row = earnings_df.iloc[-1]
        # Bayesian shrinkage: (n_stock * p_stock + prior_strength * p_global) / (n_stock + prior_strength)
        prior_strength = 20
        # Stock historical bucket stats
        earnings_explosiveness_buckets= (
            earnings_df.groupby("earnings_explosiveness_bucket")["is_extreme_reaction"]
            .agg(extreme_count="sum", event_count="count")
        )
        earnings_explosiveness_buckets["shrunk_prob"] = (
            earnings_explosiveness_buckets["extreme_count"] +
            prior_strength * bucket_stats.loc[earnings_explosiveness_buckets.index, "global_hist_prob"] # type: ignore
        ) / (
            earnings_explosiveness_buckets["event_count"] + prior_strength
        )
        earnings_explosiveness_buckets["global_hist_prob"] = bucket_stats.loc[earnings_explosiveness_buckets.index, "global_hist_prob"]
        # Lift relative to global baseline
        earnings_explosiveness_buckets["lift_vs_baseline"] = (
            earnings_explosiveness_buckets["shrunk_prob"] / P_extreme_global
        )
        # Lift relative to global same-bucket risk
        earnings_explosiveness_buckets["lift_vs_same_bucket_global"] = (
            earnings_explosiveness_buckets["shrunk_prob"] / earnings_explosiveness_buckets["global_hist_prob"]
        )

        current_bucket  = latest_row["earnings_explosiveness_bucket"]

        if type(current_bucket)!= str:
            latest_row = earnings_df.iloc[-2]
            current_bucket = latest_row["earnings_explosiveness_bucket"]

        earnings_explosiveness_score = f"{latest_row["earnings_explosiveness_score"]:.0f}"
        current_earnings_date = latest_row["earnings_date"]
        P_extreme_global = round(P_extreme_global, 3)
        current_bucket_prob = f"{earnings_explosiveness_buckets.loc[current_bucket, "shrunk_prob"]:.3f}"
        current_lift_vs_baseline = f"{earnings_explosiveness_buckets.loc[current_bucket, "lift_vs_baseline"]:.3f}"
        current_lift_vs_same_bucket_global = f"{earnings_explosiveness_buckets.loc[current_bucket, "lift_vs_same_bucket_global"]:.3f}"
        earnings_explosiveness_buckets = earnings_explosiveness_buckets.reset_index()
        # write to file
        report_txt.write(f"\n---------\n{stock}:\n")
        report_txt.write(f"Earnings Date: {current_earnings_date}\n")
        report_txt.write(f"Tail Risk Score: {earnings_explosiveness_score}\n")
        report_txt.write(f"risk_level, {current_bucket}\n")
        report_txt.write(f"base_extreme_prob, {P_extreme_global}\n")
        report_txt.write(f"hist_extreme_prob, {current_bucket_prob}\n")
        report_txt.write(f"current_lift_vs_baseline, {current_lift_vs_baseline}\n")
        report_txt.write(f"current_lift_vs_same_bucket_global, {current_lift_vs_same_bucket_global}\n")

def features_test(df):
    print("Running Feature Testing...\n--------------------")

    earnings_df = df[df["is_earnings_day"] == 1]
    earnings_df["label_3pct"] = (earnings_df["abs_reaction_3d"] >= 0.03).astype(int)
    earnings_df["label_5pct"] = (earnings_df["abs_reaction_3d"] >= 0.05).astype(int)

    earnings_df["rank"] = earnings_df.groupby("earnings_date")["risk_score"].rank(pct=True)
    earnings_df.to_csv("earnings_df_ranked.csv", index=False)

    top = earnings_df[earnings_df["rank"] >= 0.9]   # top 10%
    top["abs_reaction_3d"].mean()
    (top["abs_reaction_3d"] >= 0.05).mean()

    earnings_df["final_signal"] = earnings_df["momentum_fragility_score"]

    extreme_regime_df = earnings_df[earnings_df["earnings_explosiveness_score"] > 85].copy()  # only extreme regime
    extreme_regime_df.to_csv("extreme_regime_df.csv",index=False)

    # Testing best weights for the final risk score
    best_auc = 0
    best_w = None

    for w in np.linspace(0, 1, 21):  # 0.0 → 1.0
        score = w * earnings_df["earnings_explosiveness_score"] + (1 - w) * earnings_df["momentum_fragility_score"]
        
        data = pd.DataFrame({
            "score": score,
            "label": earnings_df["label_5pct"]
        }).dropna()
        
        auc = roc_auc_score(data["label"], data["score"])
        
        if auc > best_auc:
            best_auc = auc
            best_w = w

    print(best_auc, best_w)

    def evaluate_numeric_feature(df, feature, label_col):
        data = df[[feature, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
        
        if data[label_col].nunique() < 2:
            return None
        
        corr = data[feature].corr(data[label_col])
        
        try:
            auc = roc_auc_score(data[label_col], data[feature])
        except:
            auc = np.nan
        
        return corr, auc

    numeric_features = [
        "vol_ratio_cross_sectional_pct",
        "sector_vol_ratio_pct",
        "earnings_explosiveness_z",
        "earnings_tail_z",
        "proximity_score",
        "vol_expansion_score",
        "momentum_fragility_score",
        "earnings_explosiveness_score",
        "risk_score"
    ]

    for feature in numeric_features:
        res3 = evaluate_numeric_feature(earnings_df, feature, "label_3pct")
        res5 = evaluate_numeric_feature(earnings_df, feature, "label_5pct")
        
        print(f"{feature}")
        print(f"  3% -> corr: {res3[0]:.3f}, AUC: {res3[1]:.3f}") # type:ignore
        print(f"  5% -> corr: {res5[0]:.3f}, AUC: {res5[1]:.3f}") # type:ignore



    cat_features = [
        "vol_stress_elevated",
        "vol_stress_extreme",
        "sector_vol_stress_high",
        "momentum_pressure_regime",
        "earnings_explosiveness_bucket"
    ]
    def evaluate_categorical_feature(df, feature, label_col):
        data = df[[feature, label_col]].dropna()
        
        stats = (
            data.groupby(feature)[label_col]
            .agg(events="count", event_rate="mean")
            .sort_values("event_rate", ascending=False)
        )
        
        return stats
    
    # for feature in cat_features:
    #     print(f"\n{feature} (3%)")
    #     print(evaluate_categorical_feature(earnings_df, feature, "label_3pct"))
        
    #     print(f"\n{feature} (5%)")
    #     print(evaluate_categorical_feature(earnings_df, feature, "label_5pct"))

    def bin_analysis(df, feature, label_col, n_bins=10):
        data = df[[feature, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
        
        data["bin"] = pd.qcut(data[feature], q=n_bins, duplicates="drop")
        
        stats = (
            data.groupby("bin")[label_col]
            .agg(events="count", event_rate="mean")
        )
        
        return stats
    print(bin_analysis(earnings_df, "vol_ratio_cross_sectional_pct", "label_5pct"))
    print(bin_analysis(earnings_df, "earnings_explosiveness_score", "label_5pct"))
    print(bin_analysis(earnings_df, "risk_score", "label_5pct") )
    
# features_test(full_df)
# Test only stocks close to earnings, not just earnings day
full_df["label_3pct"] = (full_df["abs_reaction_3d"] >= 0.03).astype(int)
full_df["label_5pct"] = (full_df["abs_reaction_3d"] >= 0.05).astype(int)
pre = full_df[(full_df["days_to_earnings"] >= 1) & (full_df["days_to_earnings"] <= 10)]

pre["rank_near_earnings"] = (
    pre.groupby("date")["vol_ratio_10_to_30"]
    .rank(pct=True)
)
pre["rank_near_earnings"] = pre["rank_near_earnings"].notna()
print(pre[pre["rank_near_earnings"]])

pre_roc_score = roc_auc_score(
    pre["label_5pct"],
    pre["rank_near_earnings"]
    )


earnings_df = full_df[full_df["is_earnings_day"] == 1]
earnings_df["label_3pct"] = (earnings_df["abs_reaction_3d"] >= 0.03).astype(int)
earnings_df["label_5pct"] = (earnings_df["abs_reaction_3d"] >= 0.05).astype(int)
earnings_df["momentum_pressure_regime"] = earnings_df["momentum_pressure_regime"].notna()

earnings_df["label_8pct"] = (earnings_df["abs_reaction_3d"] >= 0.08).astype(int)

mom_roc_score = roc_auc_score(
    earnings_df["label_5pct"],
    earnings_df["momentum_fragility_score"]
    )

print("momentum_fragility_score roc_auc_score", mom_roc_score)

earnings_df["bucket"] = pd.qcut(earnings_df["momentum_fragility_score"], 10, labels=False)
# earnings_df["rank"] = earnings_df.groupby("earnings_date")["risk_score"].rank(pct=True)
earnings_df["rank"] = earnings_df.groupby("earnings_date")["momentum_fragility_score"].rank(pct=True)

print("top")
top = earnings_df[earnings_df["rank"] >= 0.9]   # top 10%
print(full_df["abs_reaction_3d"].mean())
print(top["abs_reaction_3d"].mean())
print( (top["abs_reaction_3d"] >= 0.05).mean() )

# print( earnings_df.groupby("bucket")["label_5pct"].mean() )