# pipeline/stage1.py
import warnings

from pathlib import Path

from config import (CORRECT_STOCK_COL_NAME,
                    LIST_OF_POSSIBLE_STOCK_COL_NAMES,
                    STOCK_NAMES_FILE_PATH,
                    PRICES_PROVIDER,
                    USE_CACHED_DATA_FLAG)

from data_utilities.formatting import parse_date, parse_numeric, change_column_name
from data_ingestion.fetch_stock_prices import fetch_stock_prices
from data_ingestion.fetch_earnings import fetch_earnings_dates
from data_ingestion.fetch_eps import fetch_eps
from data_ingestion.fetch_sectors import fetch_sectors_market_cap_beta
from data_utilities.clean_input import read_stocks_to_fetch
from data_utilities.merging import merge_prices_earnings_dates, merge_main_df_with_eps_df, map_sector_data_to_main_df

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
    print(f"Cached Data Usage Switch is set to: {USE_CACHED_DATA_FLAG}\n")
    if USE_CACHED_DATA_FLAG == False:
        answer = input("USE_CACHED_DATA_FLAG is switched OFF - are you sure? Y/N ")
        if answer.lower() == "n":
            print("OK, Switch USE_CACHED_DATA_FLAG ON and retry.\nExecution Stopped.")
            exit()
        if answer.lower() == "y":
            print("OK, Model won't use Cached Data. Proceeding...\n")
    warnings.filterwarnings('ignore')
    stocks = read_stocks_to_fetch(Path(STOCK_NAMES_FILE_PATH))
    print(f"{len(stocks)} Stocks to fetch.\n")
    
    stock_prices = fetch_stock_prices(provider=PRICES_PROVIDER)
    earnings_dates = fetch_earnings_dates(stocks = stocks)

    stock_prices = change_column_name(stock_prices, LIST_OF_POSSIBLE_STOCK_COL_NAMES, CORRECT_STOCK_COL_NAME )
    earnings_dates = change_column_name(earnings_dates, LIST_OF_POSSIBLE_STOCK_COL_NAMES, CORRECT_STOCK_COL_NAME )
    
    # Sort to prep for merge_asof
    stock_prices["date"] = parse_date(stock_prices["date"])
    earnings_dates["earnings_date"] = parse_date(earnings_dates["earnings_date"])

    stock_prices = stock_prices.sort_values("date")
    earnings_dates = earnings_dates.sort_values("earnings_date")
    
    

    df = merge_prices_earnings_dates(stock_prices, earnings_dates) # df that holds stock prices, earnings dates, EPS data merged

    
    eps_data = fetch_eps(stocks)
    # TODO: EPS is fetch but not merged now. ignored for as Im focusing on pre-earnings features
    #df = merge_main_df_with_eps_df(df, eps_data)

    sector_market_cap_beta_df = fetch_sectors_market_cap_beta()
    
    df = map_sector_data_to_main_df(df, sector_market_cap_beta_df)

    # Sort, make sure "price" is numeric, make sure dates are datetime just in case
    df = df.sort_values(["stock", "date"]).reset_index(drop=True)

    df["date"] = parse_date(df["date"])
    df["earnings_date"] = parse_date(df["earnings_date"])
    df["price"] = parse_numeric(df["price"])

    return df
    