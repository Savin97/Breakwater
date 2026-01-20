import pandas as pd

def merge_prices_earnings_dates(stock_prices, earnings_dates):
    prices_earnings_dates_df = ( pd.merge_asof(
                    stock_prices, earnings_dates, left_on='date',
                    right_on = 'earnings_date', by = "stock", direction="forward") )
    # Merge the result with EPS data
    return prices_earnings_dates_df
