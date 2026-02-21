# main.py
from pipeline.pipeline import run_pipeline
from db.db_main import db_main
from report.report_builder import generate_report
from data_ingestion.api_functions import get_earnings_data_from_api

def main():
    print("--------------------\nRunning pipeline...\n--------------------\n")

    def get_earnings_dates_yf(ticker: str, limit : int = 12):
        import pandas as pd
        import yfinance as yf
        t = yf.Ticker(ticker)
        t = yf.Ticker("wmb")
        df = t.get_earnings_dates(limit=limit)
        if df is None or df.empty:
            return pd.DataFrame(columns=[
                "earnings_date",
                "eps_estimate",
                "reported_eps",
                "surprise_pct"
            ])

        df = df.reset_index()

        df = df.rename(columns={
            "Earnings Date": "earnings_date",
            "EPS Estimate": "eps_estimate",
            "Reported EPS": "reported_eps",
            "Surprise(%)": "surprise_pct"
        })

        #Keep only relevant columns
        df = df[[
            "earnings_date",
            "eps_estimate",
            "reported_eps",
            "surprise_pct"
        ]]

        df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.date
        df.to_csv("yf_test.csv",index=False)

    db_main()
    #generate_report()
    #run_pipeline()
    print("Pipeline execution completed.\n--------------------")

if __name__ == "__main__":
    main()
