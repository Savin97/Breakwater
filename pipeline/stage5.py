# pipeline/stage5.py
from report.report_builder import generate_report
def stage5(df):
    print("Stage 5 - Generating Report...")
    stocks_to_report_for = ["AAPL","AMD"]
    for stock in stocks_to_report_for:
        stock_df = df[df["stock"]==stock]
        data_for_report = {
            "earnings_date": df["earnings_date"],
            "risk_level": "mid",
            "risk_score": df["timing_danger"],}
        generate_report(stock, data_for_report)
    return df