import pandas as pd

def merge_prices_earnings_dates_eps(stock_prices, earnings_dates):
    prices_earnings_dates_df = stock_prices.merge(earnings_dates, left_on=['symbol', 'date'],right_on = ['symbol', 'earnings_date'], how='left')
    # Merge the result with EPS data
    return prices_earnings_dates_df
