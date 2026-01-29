import pandas as pd

heavy_prices = pd.read_csv("data/Copy of 2005 onward cached data/stock_prices.csv")
heavy_dates = pd.read_csv("data/Copy of 2005 onward cached data/earnings_dates.csv")
heavy_prices["date"] = pd.to_datetime(heavy_prices["date"])
heavy_dates["earnings_date"] = pd.to_datetime(heavy_dates["earnings_date"])

heavy_prices = heavy_prices[heavy_prices["date"] > "2021-01-01"]
heavy_dates = heavy_dates[heavy_dates["earnings_date"] > "2021-01-01"]

heavy_prices.to_csv("heavy_prices",index=False)