feature definition explanation
stock
date
earnings_date
60d_drift |60 day stock drift, daily rolling 60 mean | AVG of past 60 daily returns (with .shift(1)) ; expectations about this company
vol_10,30 | 10,30 day Volatility | STD of past 10/30 daily returns (with .shift(1))
mom_5,20 | 5,20 day Momentum | Sum of past 5/20 daily returns (with .shift(1))

Stock Drift |  | Capital flow / macro / theme pressure affecting all stocks in that bucket Earnings reactions are amplified when both Stock drift and Sector drift align.