import pandas as pd
import yfinance as yf
import datetime
#from data_utilities.helper_funcs import read_stocks_to_fetch

def next_earnings_from_calendar(calendar_df: pd.DataFrame, sp500_symbols: set) -> pd.DataFrame:
    df = calendar_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["symbol"].isin(sp500_symbols)]
    out = (df.groupby("symbol", as_index=False)["date"]
             .min()
             .rename(columns={"date": "earnings_date"}))
    return out

#stocks = read_stocks_to_fetch()
today = pd.Timestamp.today(tz="America/New_York").normalize()
print(type(today))
# print(today)
ticker = yf.Ticker("AAPL")
df = ticker.get_earnings_dates(limit=1, offset=0)

if df is not None:
    df= df.reset_index()
    print(df.columns)
    edates = pd.to_datetime(df["Earnings Date"]).dt.normalize()
    print(edates.dtype)
else:
    raise ValueError("Nothng from yfinance")

for date in edates:
    date = pd.to_datetime(date)
    if date >= today:
        print(edates)