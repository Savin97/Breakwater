# pipeline/stage1.py
import warnings

from config import (CORRECT_STOCK_COL_NAME,
                    LIST_OF_POSSIBLE_STOCK_COL_NAMES,
                    PRICES_PROVIDER,)

from data_utilities.formatting import parse_date, parse_numeric, change_column_name
from data_ingestion.fetch_stock_prices import fetch_stock_prices
from data_ingestion.fetch_earnings import fetch_earnings_dates
from data_ingestion.fetch_eps import fetch_eps
from data_ingestion.fetch_sectors import fetch_sectors_market_cap_beta
from data_utilities.merging import (merge_prices_earnings_dates, 
                                    merge_main_df_with_eps_df, 
                                    map_sector_data_to_main_df)
from data_utilities.helper_funcs import directory_checks

def stage1():
    """
        First stage of the pipeline - Data Ingestion:
        Fetch historical stock prices,
        Earnings dates,
        EPS data, 
        Sector & Sub-sector data, 
        Market cap, 
        Beta

        Merge into one DF and return it.
    """
    directory_checks()   
    warnings.filterwarnings('ignore')

    # def check_stock_prices():
    #     from pathlib import Path
    #     pass
    #     # if Path("data/stock_prices.csv").exists():
    #     """
    #         check_prices_file()
    #         if the file already exists, check if it has
    #             1. the correct columns: stock, date, price
    #             2. data from STOCKS_START_DATE to STOCKS_END_DATE
    #             if it has full data - pd.read_csv and return 
    #             if it has partial data:
    #                 - if it has the correct columns but only partial date range, fetch the rest and append
    #                 - if it doesn't -  delete and re-fetch
            

    #     """
    #     # else:
    #         # if the file doesn't exist - fetch new and save 
    #         # stock_prices = fetch_stock_prices(provider=PRICES_PROVIDER)

    # def check_earnings_dates():
    #     from pathlib import Path
    #     pass
    #     # if Path("data/earnings_dates.csv").exists():
    #     """
    #         check_earnings_file()
    #         if the file already exists, check if it has
    #             1. the correct columns: stock, date, earnings_date
    #             2. data from STOCKS_START_DATE to STOCKS_END_DATE
    #             if it has full data - pd.read_csv and return 
    #             if it has partial data:
    #                 - if it has the correct columns but only partial date range, fetch the rest and append
    #                 - if it doesn't -  delete and re-fetch
    #     """
    #     # else:
    #         # if the file doesn't exist - fetch new and save 
    #         # earnings_dates = fetch_earnings_dates()

    # def check_sector_data():
    #     from pathlib import Path
    #     pass
    #     # if Path("data/sector_data.csv").exists():
    #     """
    #         check_sector_data_file()
    #         if the file already exists, check if it has
    #             1. the correct columns: stock, date, sector, sub_sector, market_cap, beta
    #             2. data from STOCKS_START_DATE to STOCKS_END_DATE
    #             if it has full data - pd.read_csv and return 
    #             if it has partial data:
    #                 - if it has the correct columns but only partial date range, fetch the rest and append
    #                 - if it doesn't -  delete and re-fetch
    #     """
    #     # else:
    #         # if the file doesn't exist - fetch new and save 
    #         # sector_market_cap_beta_df = fetch_sector_data()


    stock_prices = fetch_stock_prices(provider=PRICES_PROVIDER)
    earnings_dates = fetch_earnings_dates()

    stock_prices = change_column_name(stock_prices, LIST_OF_POSSIBLE_STOCK_COL_NAMES, CORRECT_STOCK_COL_NAME )
    earnings_dates = change_column_name(earnings_dates, LIST_OF_POSSIBLE_STOCK_COL_NAMES, CORRECT_STOCK_COL_NAME )
    
    # Sort to prep for merge_asof
    stock_prices["date"] = parse_date(stock_prices["date"])
    earnings_dates["earnings_date"] = parse_date(earnings_dates["earnings_date"])

    stock_prices = stock_prices.sort_values("date")
    earnings_dates = earnings_dates.sort_values("earnings_date")
    
    df = merge_prices_earnings_dates(stock_prices, earnings_dates) # df that holds stock prices, earnings dates, EPS data merged

    eps_data = fetch_eps()
    # TODO: EPS is fetched but not merged now. ignored for as Im focusing on pre-earnings features
    #df = merge_main_df_with_eps_df(df, eps_data)

    sector_market_cap_beta_df = fetch_sectors_market_cap_beta()
    
    df = map_sector_data_to_main_df(df, sector_market_cap_beta_df)

    # Sort, make sure "price" is numeric, make sure dates are datetime just in case
    df = df.sort_values(["stock", "date"]).reset_index(drop=True)
    df["date"] = parse_date(df["date"])
    df["earnings_date"] = parse_date(df["earnings_date"])
    df["price"] = parse_numeric(df["price"])
    
    return df

def janky_solution():
    from config import partial_tickers_to_fetch
    import pandas as pd
    prices = pd.read_csv("data/stock_prices.csv")
    earnings = pd.read_csv("data/earnings_dates.csv")
    prices = prices[prices["date"]>= "2016-01-01"]
    earnings = earnings[earnings["earnings_date"]>= "2016-01-01"]

    prices = prices[prices["stock"].isin(partial_tickers_to_fetch)]
    earnings = earnings[earnings["stock"].isin(partial_tickers_to_fetch)]
    prices.to_csv("data/stock_prices.csv",index=False)
    earnings.to_csv("data/earnings_dates.csv",index=False)