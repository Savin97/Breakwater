# main.py
from pipeline.pipeline import run_pipeline

def main():
    print("--------------------\nRunning pipeline...\n--------------------\n")
    # import pandas as pd
    # prices = pd.read_csv("data/stock_prices.csv")
    # earnings = pd.read_csv("data/earnings_dates.csv")
    # prices = prices[prices["date"]>= "2016-01-01"]
    # earnings = earnings[earnings["earnings_date"]>= "2016-01-01"]
    # prices.to_csv("data/stock_prices.csv",index=False)
    # earnings.to_csv("data/earnings_dates.csv",index=False)

    run_pipeline()    
    print("Pipeline execution completed.\n--------------------")
    

if __name__ == "__main__":
    main()
