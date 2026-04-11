# pipeline/stage5.py
from report.report_builder import generate_report
import pandas as pd
def stage5(df): # stage5(df);
    print("--------------------\nStage 5 - Generating Report...")
    df = pd.read_parquet("output/full_df.parquet")
    stock_list = pd.read_csv("data/stock_list.csv")
    first_100_stocks = stock_list.iloc[1:51,0]
    print(first_100_stocks.iloc[0])
    stocks_to_report_for = []
    global_earnings_df = df[df["is_earnings_day"] == 1].copy()
    P_extreme_global  = global_earnings_df["is_extreme_reaction"].mean()
    P_extreme_given_bucket = (
        global_earnings_df.groupby("earnings_explosiveness_bucket")["is_extreme_reaction"]
        .mean()
    )
    bucket_stats = pd.DataFrame({
        "global_hist_prob": P_extreme_given_bucket,
        "global_risk_lift_vs_baseline": P_extreme_given_bucket / P_extreme_global
    })

    global_earnings_df = global_earnings_df[
        global_earnings_df["stock"].isin(stocks_to_report_for)
    ]
    #global_earnings_df.to_csv("global_earnings_df.csv",index=False)
    report_txt = open("report_txt.txt", "w")
    for stock in first_100_stocks:
        print(f"\n---------\n{stock}:")
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
            prior_strength * bucket_stats.loc[earnings_explosiveness_buckets.index, "global_hist_prob"]
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
        # print("bucket_table", earnings_explosiveness_buckets")
  
        data_for_report = {
            "earnings_date": current_earnings_date,
            "risk_level": current_bucket,
            "risk_score": earnings_explosiveness_score,
            "base_extreme_prob": P_extreme_global,
            "hist_extreme_prob": current_bucket_prob,
            "current_lift_vs_baseline": current_lift_vs_baseline,
            "current_lift_vs_same_bucket_global": current_lift_vs_same_bucket_global,
            "bucket_table": earnings_explosiveness_buckets
            }
        #generate_report(stock, data_for_report)
    print("Stage 5 DONE")
    return df