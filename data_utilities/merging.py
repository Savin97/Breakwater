import pandas as pd
from data_utilities.formatting import parse_date

def merge_prices_earnings_dates(stock_prices, earnings_dates):
    merged_df = ( pd.merge_asof(
                    stock_prices, earnings_dates, left_on='date',
                    right_on = 'earnings_date', by = "stock", direction="forward") )
    # Merge the result with EPS data
    return merged_df

def merge_main_df_with_eps_df(main_df,eps_df):
    # Needs to merge main_df with eps_df on left = "date" and right = "reported_date"
    # Needs to bring in columns from eps_df of [reported_eps	estimated_eps	surprise_percentage]
    main_df["date"] = parse_date(main_df["date"])
    eps_df["reported_date"] = parse_date(eps_df["reported_date"])
    merged_df = main_df.merge(eps_df, 
                              left_on = ["stock", "date"], 
                              right_on = ["symbol", "reported_date"], how="left")
    # Drop duplicate columns
    merged_df = merged_df.drop(columns= ["symbol","reported_date", "fiscal_date"])
    
    return merged_df

def merge_main_df_with_sectors_df(main_df,sector_df):
    pass
