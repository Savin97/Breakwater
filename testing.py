# testing.py
import pandas as pd, warnings, matplotlib.pyplot as plt, numpy as np
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
    # pressure -> crowding / exhaustion
    # bias -> one-sided positioning
    # sector drift -> late-cycle momentum

    bias_scale = df["directional_bias"].abs().quantile(0.90)

    m1 = df["momentum_pressure_regime"].map(PRESSURE_MAP).fillna(0)
    # m2: this achieves: 90% of observations live in [0,1),Top 10% saturate at 1, 
    # No arbitrary magic number, Stable across stocks and time
    m2 = np.clip(np.abs(df["directional_bias"].fillna(0)) / bias_scale, 0, 1) 
    m3 = np.clip(np.abs(df["sector_drift_60d"].fillna(0)) / 0.10, 0, 1)

    #m2 = (df["directional_bias"].abs() / bias_scale).clip(0, 1)
    base = (
        0.45 * m1 +   # positioning pressure
        0.35 * m3 +   # directional skew
        0.20 * m2    # sector trend maturity
    )

    momentum_fragility_score = 100 * np.clip(base, 0, 1)
    return momentum_fragility_score

def engineer_momentum_fragility_score(input_df):
    df = input_df.copy()
    df["momentum_fragility_score"] = score_momentum_fragility(df)
    return df
    
def engineer_earnings_explosiveness_score(input_df):
    df = input_df.copy()
    df["earnings_explosiveness_score"] = score_earnings_explosiveness(df)
    return df
warnings.filterwarnings('ignore')


# ---------------------------------------------
# TESTING
# ---------------------------------------------


print("---------------------------------------------\nTESTING\n---------------------------------------------")
original_df = pd.read_parquet("output/full_df.parquet")
df = original_df.copy()
label_col = "is_extreme_reaction"
# Risk score
exp = score_earnings_explosiveness(df)
mom = score_momentum_fragility(df)

high_exp = exp >= exp.quantile(0.90)
risk_score = exp.copy()
risk_score += 0.5 * mom * high_exp
risk_score = np.clip(risk_score,0,100)
high_risk = (
    (exp >= exp.quantile(0.90))
    &
    (mom >= mom.quantile(0.70))
)
high_exp = exp >= exp.quantile(0.90)
high_mom = mom >= mom.quantile(0.70)

group = high_exp & high_mom
global_extreme_rate = df[label_col].mean()
event_rate = df.loc[group, "is_extreme_reaction"].mean()
lift = event_rate / global_extreme_rate
count = group.sum()

exit()

df["risk_score"] = risk_score

global_earnings_df = df[df["is_earnings_day"] == 1].copy()
features = ["vol_ratio_cross_sectional_pct", "earnings_explosiveness_score", "sector_vol_ratio_pct", "vol_expansion_score", "momentum_fragility_score",  "risk_score"]
features = ["earnings_explosiveness_score", "earnings_explosiveness_score", "risk_score"]
for feature_col in features:
    print(f"---------------------------------------------\nNow evaluating feature {feature_col}:\n---------------------------------------------")
    data = global_earnings_df[[feature_col, label_col]].copy()
    n_bins = 10
    data = data.replace([np.inf, -np.inf], np.nan).dropna() 

    # Bin into quantiles (deciles)
    data["bin"] = pd.qcut(
        data[feature_col],
        q=n_bins,
        duplicates="drop"
    )

    # Global event rate
    global_extreme_rate = data[label_col].mean()

    # Aggregate stats per bin
    stats = (
        data.groupby("bin")[label_col]
        .agg(
            events="count",
            event_rate="mean"
        )
    )

    # Risk lift
    stats["risk_lift"] = stats["event_rate"] / global_extreme_rate
    stats = stats.reset_index()

    print("Global rate:", global_extreme_rate)
    print(stats)

    # ---- PLOT ----
    plt.figure(figsize=(10, 6))

    plt.plot(
        stats.index + 1, 
        stats["risk_lift"],
        marker="o"
    )
    plt.xticks(stats.index + 1, stats["bin"].astype(str), rotation=45)
    plt.xlabel(f"{feature_col} Feature Decile")
    plt.ylabel("Risk Lift")
    plt.title(f"Risk Lift by {feature_col} Decile")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"output/testing/{feature_col}.png")


# P_extreme_global  = global_earnings_df["is_extreme_reaction"].mean()
# P_extreme_given_bucket = (
#     global_earnings_df.groupby("timing_danger_bucket")["is_extreme_reaction"]
#     .mean()
# )
# bucket_stats = pd.DataFrame({
#     "hist_prob": P_extreme_given_bucket,
#     "risk_lift": P_extreme_given_bucket / P_extreme_global
# })
# bins = [0, 5, 10, 20, 30, 50, 100]
# labels = ["0-5", "5-10", "10-20", "20-30", "30", "50+"]

# stocks_to_report_for = ["A", "AAPL" ,"ABBV" ,"ABNB" ,"ABT" ,"ACGL" ,"ACN" ,"ADBE" ,"ADI","AMD"]
# for stock in stocks_to_report_for:
#     print(f"\n---------------------------------------------\n{stock}:\n---------------------------------------------\n")
#     stock_df = df[df["stock"] == stock]
#     earnings_df = stock_df[stock_df["is_earnings_day"] == 1]

#     # Bayesian shrinkage: (n_stock * p_stock + prior_strength * p_global) / (n_stock + prior_strength)
#     prior_strength = 20
#     hist_extreme_prob = (
#         earnings_df.groupby("timing_danger_bucket")["is_extreme_reaction"]
#         .agg(["sum","count"])
#     )
#     hist_extreme_prob["prob"] = (
#         hist_extreme_prob["sum"] +
#         prior_strength * bucket_stats.loc[hist_extreme_prob.index,"hist_prob"]
#         ) / (
#         hist_extreme_prob["count"] + prior_strength
#     )
#     bucket_prob = bucket_stats["hist_prob"]

#     risk_lift = hist_extreme_prob["prob"] / bucket_prob.loc[hist_extreme_prob.index]
#     counts = earnings_df.groupby("timing_danger_bucket").size()

#     extreme_counts = (
#         earnings_df.groupby("timing_danger_bucket")["is_extreme_reaction"]
#         .sum()
#     )
#     print(pd.DataFrame({
#         "events": counts,
#         "extreme": extreme_counts,
#         "prob": hist_extreme_prob["prob"],
#         "risk_lift": risk_lift
#     }))




# # timing_danger
# # Decile bins
# global_earnings_df["danger_bin"] = pd.qcut(
#     global_earnings_df["timing_danger"],
#     q=10
# )

# # Aggregate
# bin_stats = (
#     global_earnings_df
#     .groupby("danger_bin", observed=False)["is_extreme_reaction"]
#     .agg(["count", "mean"])
#     .rename(columns={"mean": "extreme_rate"})
#     .reset_index()
# )

# # Midpoint of each bin (for smooth x-axis)
# bin_stats["bin_mid"] = bin_stats["danger_bin"].apply(
#     lambda x: (x.left + x.right) / 2
# )

# # Compute risk lift
# bin_stats["risk_lift"] = bin_stats["extreme_rate"] / P_extreme_global

# # ---- PLOT ----
# plt.figure(figsize=(10, 6))

# plt.plot(
#     bin_stats["bin_mid"],
#     bin_stats["risk_lift"],
#     marker="o"
# )

# plt.xlabel("timing_danger")
# plt.ylabel("Risk Lift (vs Global Baseline)")
# plt.title("Risk Lift vs timing_danger (Deciles)")

# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# # earnings_explosiveness
# # Decile bins
# global_earnings_df["danger_bin"] = pd.qcut(
#     global_earnings_df["earnings_explosiveness_score"],
#     q=10
# )

# # Aggregate
# bin_stats = (
#     global_earnings_df
#     .groupby("danger_bin", observed=False)["is_extreme_reaction"]
#     .agg(["count", "mean"])
#     .rename(columns={"mean": "extreme_rate"})
#     .reset_index()
# )

# # Midpoint of each bin (for smooth x-axis)
# bin_stats["bin_mid"] = bin_stats["danger_bin"].apply(
#     lambda x: (x.left + x.right) / 2
# )

# # Compute risk lift
# bin_stats["risk_lift"] = bin_stats["extreme_rate"] / P_extreme_global

# # ---- PLOT ----
# plt.figure(figsize=(10, 6))

# plt.plot(
#     bin_stats["bin_mid"],
#     bin_stats["risk_lift"],
#     marker="o"
# )

# plt.xlabel("earnings_explosiveness_score")
# plt.ylabel("Risk Lift (vs Global Baseline)")
# plt.title("Risk Lift vs earnings_explosiveness_score (Deciles)")

# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()