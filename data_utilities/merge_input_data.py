import pandas as pd

def last_earnings_date(stock_prices, earnings_dates):
    last_earnings = stock_prices.merge_asof(earnings_dates, left_on = "date", right_on = "earnings_date")

def merge_prices_earnings_dates(stock_prices, earnings_dates):

    prices_earnings_dates_df = stock_prices.merge(earnings_dates, left_on=['symbol', 'date'],right_on = ['stock', 'earnings_date'], how='left')
    # Merge the result with EPS data
    return prices_earnings_dates_df
