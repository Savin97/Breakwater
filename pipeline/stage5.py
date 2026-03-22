# pipeline/stage5.py
from report.report_builder import generate_report
import pandas as pd
def stage5(df): # stage5(df);
    print("--------------------\nStage 5 - Generating Report...")

    stocks_to_report_for = ["A", "AAPL" ,"ABBV" ,"ABNB" ,"ABT" ,"ACGL" ,"ACN" ,"ADBE" ,"ADI","AMD"]
    global_earnings_df = df[df["is_earnings_day"] == 1].copy()
    P_extreme_global  = global_earnings_df["is_extreme_reaction"].mean()
    P_extreme_given_bucket = (
        global_earnings_df.groupby("timing_danger_bucket")["is_extreme_reaction"]
        .mean()
    )
    bucket_stats = pd.DataFrame({
        "hist_prob": P_extreme_given_bucket,
        "risk_lift": P_extreme_given_bucket / P_extreme_global
    })

    for stock in stocks_to_report_for:
        print(f"\n---------\n{stock}:")
        stock_df = df[df["stock"] == stock]
        earnings_df = stock_df[stock_df["is_earnings_day"] == 1]

        # Bayesian shrinkage: (n_stock * p_stock + prior_strength * p_global) / (n_stock + prior_strength)
        prior_strength = 20
        hist_extreme_prob = (
            earnings_df.groupby("timing_danger_bucket")["is_extreme_reaction"]
            .agg(["sum","count"])
        )

        hist_extreme_prob["prob"] = (
            hist_extreme_prob["sum"] +
            prior_strength * bucket_stats.loc[hist_extreme_prob.index,"hist_prob"]
            ) / (
            hist_extreme_prob["count"] + prior_strength
        )
        bucket_prob = bucket_stats["hist_prob"]
        risk_lift = hist_extreme_prob["prob"] / bucket_prob.loc[hist_extreme_prob.index]

        last_earnings_date = stock_df[stock_df["earnings_date"].notna()]["earnings_date"].iloc[-1].date()
        timing_danger = f"{stock_df.loc[stock_df['earnings_date'].notna(), 'timing_danger'].iloc[-1]:.0f}"
        data_for_report = {
            "earnings_date": last_earnings_date,
            "risk_level": "mid",
            "risk_score": timing_danger,
            "base_extreme_prob": P_extreme_global,
            "hist_extreme_prob": hist_extreme_prob,
            "risk_lift": risk_lift
            }
        generate_report(stock, data_for_report)
    print("Stage 5 DONE")
    return df