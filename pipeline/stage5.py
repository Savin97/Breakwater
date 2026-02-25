# pipeline/stage5.py
from report.report_builder import generate_report
def stage5(df):
    print("--------------------\nStage 5 - Generating Report...")
    stocks_to_report_for = ["AAPL","AMD"]
    for stock in stocks_to_report_for:
        stock_df = df[df["stock"]==stock]
        last_earnings_date = stock_df[stock_df["earnings_date"].notna()]["earnings_date"].iloc[-1].date()
        timing_danger_score = f"{stock_df.loc[stock_df['earnings_date'].notna(), 'timing_danger_score'].iloc[-1]:.0f}"
        data_for_report = {
            "earnings_date": last_earnings_date,
            "risk_level": "mid",
            "risk_score": timing_danger_score,
            "hist_xtreme_prob": "hist_xtreme_prob",
            "base_xtreme_prob": "base_xtreme_prob",
            "risk_lift": "risk_lift"
            }
        generate_report(stock, data_for_report)
    print("Stage 5 Done.")
    return df