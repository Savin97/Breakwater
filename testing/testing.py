import pandas as pd, numpy as np, warnings
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

df = pd.read_parquet("output/full_df.parquet")

# ── Grid search: normalization denominator for earnings_explosiveness_score ──
# Recomputes gated_explosiveness_score inline for each candidate denominator.
# No need to re-run main.py — raw features are already in the parquet.
# Key metric: avg OOS Pearson correlation (walk-forward, year-by-year).

def score_at_denom(df, denom):
    epsilon = 1e-6
    vol = np.maximum(df["vol_30d"], epsilon)
    p75 = df["abs_reaction_p75_rolling"].fillna(df["abs_reaction_p75"])
    e1 = (df["earnings_explosiveness_z"].fillna(0) / denom).clip(0, 1)
    e2 = (p75 / vol / denom).clip(0, 1)
    e3 = (p75 / 0.12).clip(0, 1)
    e4 = np.clip(df["reaction_entropy"], 0, 1)
    exp = 100 * np.clip(0.35 * e2 + 0.30 * e1 + 0.25 * e3 + 0.10 * e4, 0, 1)
    vol_gate = (
        1.0
        + 0.4 * (df["stock_vs_sector_vol"].fillna(1) > 1).astype(float)
        + 0.3 * df["vol_stress_extreme"].fillna(0).astype(float)
    )
    return (exp * vol_gate).clip(0, 100)


def avg_oos_corr(df, score_series, date_col="date", target_col="abs_reaction_3d"):
    d = df[df["is_earnings_day"] == 1][[date_col, target_col]].copy()
    d["score"] = score_series.loc[d.index]
    d = d.dropna()
    d[date_col] = pd.to_datetime(d[date_col])
    corrs = []
    for y in sorted(d[date_col].dt.year.unique())[1:]:
        train = d[d[date_col] < pd.Timestamp(f"{y}-01-01")]
        test  = d[(d[date_col] >= pd.Timestamp(f"{y}-01-01")) & (d[date_col] < pd.Timestamp(f"{y+1}-01-01"))]
        if len(train) < 500 or len(test) < 100:
            continue
        lo, hi = train["score"].min(), train["score"].max()
        if hi == lo:
            continue
        test_score = (test["score"] - lo) / (hi - lo)
        corrs.append(test_score.corr(test[target_col]))
    return np.mean(corrs) if corrs else np.nan


spot_stocks = ["AAPL", "TSLA", "NVDA", "MSFT"]
results = []

for denom in [4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20,25,30,35,40]:
    df["_gated"] = score_at_denom(df, denom)
    corr = avg_oos_corr(df, df["_gated"])
    latest = {}
    for s in spot_stocks:
        sdf = df[(df["stock"] == s) & (df["is_earnings_day"] == 1)]
        latest[s] = f"{sdf['_gated'].iloc[-1]:.0f}" if not sdf.empty else "n/a"
    results.append({"denom": denom, "avg_oos_corr": round(corr, 4), **latest})

df.drop(columns=["_gated"], inplace=True)

print(pd.DataFrame(results).to_string(index=False))

# ── Test: simplified score (no vol-normalized components) ──
def score_simplified(df, e3_w=0.85, e4_w=0.15):
    p75 = df["abs_reaction_p75_rolling"].fillna(df["abs_reaction_p75"])
    e3 = (p75 / 0.12).clip(0, 1)
    e4 = np.clip(df["reaction_entropy"], 0, 1)
    exp = 100 * np.clip(e3_w * e3 + e4_w * e4, 0, 1)
    vol_gate = (
        1.0
        + 0.4 * (df["stock_vs_sector_vol"].fillna(1) > 1).astype(float)
        + 0.3 * df["vol_stress_extreme"].fillna(0).astype(float)
    )
    return (exp * vol_gate).clip(0, 100)

df["_simplified"] = score_simplified(df)
corr_s = avg_oos_corr(df, df["_simplified"])
simple_latest = {}
for s in spot_stocks:
    sdf = df[(df["stock"] == s) & (df["is_earnings_day"] == 1)]
    simple_latest[s] = f"{sdf['_simplified'].iloc[-1]:.0f}" if not sdf.empty else "n/a"
df.drop(columns=["_simplified"], inplace=True)
print(f"\nSimplified (no e1/e2): avg_oos_corr={round(corr_s,4)}  {simple_latest}")


exit()
##################################
# FREE-FORM TESTING 
##################################

df = df.drop(columns=["momentum_fragility_score", "risk_score", "momentum_pressure_regime"] )
df["label_3pct"] = (df["abs_reaction_3d"] >= 0.03).astype(int)
df["label_5pct"] = (df["abs_reaction_3d"] >= 0.05).astype(int)
pre_earnings = df[(df["days_to_earnings"] >= 1) & (df["days_to_earnings"] <= 10)]
earnings_df = df[df["is_earnings_day"] == 1].copy()

##################################
# Backtesting features
##################################
print("##################################\nBacktesting features\n##################################")
feature = "earnings_explosiveness_score"

test_score_df = df[["stock", "date","earnings_date","abs_reaction_3d", feature]].dropna().copy()
pre_2015 = test_score_df[test_score_df["date"] < "2015-01-01"].copy()
post_2015 = test_score_df[test_score_df["date"] >= "2015-01-01"].copy()

def normalize_with_train(s, train_min, train_max):
    denom = train_max - train_min
    if denom == 0:
        return pd.Series(50, index=s.index)
    return 100 * (s - train_min) / denom

train_min = pre_2015[feature].min()
train_max = pre_2015[feature].max()
pre_2015["score_oos"] = normalize_with_train(
    pre_2015[feature], train_min, train_max
)

post_2015["score_oos"] = normalize_with_train(
    post_2015[feature], train_min, train_max
).clip(0, 100)
pre_corr = pre_2015[["score_oos", "abs_reaction_3d"]].corr().iloc[0,1]
print("Train corr:", pre_corr)

pre_2015["bucket"] = pd.qcut(pre_2015["score_oos"], q=10, labels=False)
print(pre_2015.groupby("bucket")["abs_reaction_3d"].mean())
train_edges = pd.qcut(
    pre_2015["score_oos"],
    q=10,
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
    
    report_txt = open("output/report_txt.txt", "w")
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

    # print(best_auc, best_w)

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
    print(bin_analysis(earnings_df, "momentum_fragility_score", "label_5pct"))
    # print(bin_analysis(earnings_df, "earnings_explosiveness_score", "label_5pct"))
    # print(bin_analysis(earnings_df, "risk_score", "label_5pct") )


earnings_df["fragility_pct"] = (
    earnings_df.groupby("date")["momentum_fragility_score"]
    .rank(pct=True)
)

earnings_df["fragility_display_score"] = 100 * earnings_df["fragility_pct"]
earnings_df["bucket"] = pd.qcut(earnings_df["momentum_fragility_score"], 10, labels=False)
earnings_df["rank"] = earnings_df.groupby("earnings_date")["momentum_fragility_score"].rank(pct=True)
top = earnings_df[earnings_df["rank"] >= 0.9]   # top 10%
# print( earnings_df.groupby("bucket")["label_5pct"].mean() )
# print(earnings_df["fragility_pct"])
