# pipeline/stage1.py
import warnings

from config import (CORRECT_STOCK_COL_NAME,
                    LIST_OF_POSSIBLE_STOCK_COL_NAMES,
                    PRICES_PROVIDER)
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

        DuckDB version:
        1. Create DB/Make sure it exists.
        2. Create prices, earnings, sector tables / make sure they exist.
        3. Update tables or choose to leave them as-is (introduce a switch for this)
        4. Import tables as needed for the pipeline. For example, if start date is set to 2020-01-01 until today, then
        the pipeline should take what it needs from the database and put it in a pandas df.
        5. Merge DFs to one df.
        6. The result should be a df that has cols
        stock price date earnings_date estimated_eps reported_eps surprise_percentage
    """
    directory_checks()   
    warnings.filterwarnings('ignore')

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