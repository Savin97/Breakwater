import pandas as pd

def last_earnings_date(stock_prices, earnings_dates):
    last_earnings = stock_prices.merge_asof(earnings_dates, left_on = "date", right_on = "earnings_date")

def merge_prices_earnings_dates(stock_prices, earnings_dates):

    prices_earnings_dates_df = ( pd.merge_asof(
                    stock_prices, earnings_dates, left_on='date',
                    right_on = 'earnings_date', by = "stock", direction="forward") )
    # Merge the result with EPS data
    return prices_earnings_dates_df
