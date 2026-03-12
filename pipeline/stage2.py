# pipeline/stage2.py
import duckdb
from config import (
    CORRECT_STOCK_COL_NAME,
    LIST_OF_POSSIBLE_STOCK_COL_NAMES,
    PRICES_PROVIDER,
    DB_PATH)
from data_ingestion.data_utilities import (
    parse_date, parse_numeric, 
    change_column_name,
    merge_prices_earnings_dates,
    map_sector_data_to_main_df)
def stage2():
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
        1. Import tables as needed for the pipeline. For example, if start date is set to 2020-01-01 until today, then
        the pipeline should take what it needs from the database and put it in a pandas df.
        2. Merge DFs to one df.
        3. The result should be a df that has cols
        stock price date earnings_date estimated_eps reported_eps surprise_percentage
    """
    print("--------------------\nStage 2 - Data Ingestion...")
    con = duckdb.connect(DB_PATH)
    prices_df = con.execute("SELECT stock,price,date FROM prices ORDER BY stock,date").fetch_df()
    earnings_df = con.execute("SELECT stock,earnings_date FROM earnings ORDER BY stock,earnings_date").fetch_df()
    # stock_data_df = con.execute("SELECT * FROM stock_data ORDER BY stock").fetch_df()
    prices_df["date"] = parse_date(prices_df["date"])
    earnings_df["earnings_date"] = parse_date(earnings_df["earnings_date"])
    prices_df = prices_df.sort_values("date")
    earnings_df = earnings_df.sort_values("earnings_date")
    df = merge_prices_earnings_dates(prices_df, earnings_df) # df that holds stock prices, earnings dates
    #sector_market_cap_beta_df = fetch_sectors_market_cap_beta()
    import pandas as pd
    sector_df = pd.read_csv("sp500_data.csv")
    sector_df = sector_df[["stock","name","sector","sub_sector"]]
    df = map_sector_data_to_main_df(df, sector_df)

    # Sort, make sure "price" is numeric, make sure dates are datetime just in case
    df = df.sort_values(["stock", "date"]).reset_index(drop=True)
    df["date"] = parse_date(df["date"])
    df["earnings_date"] = parse_date(df["earnings_date"])
    df["price"] = parse_numeric(df["price"])
    print("Stage 2 DONE")
    return df

    # stock_prices = fetch_stock_prices(provider=PRICES_PROVIDER)
    # earnings_dates = fetch_earnings_dates()

    # stock_prices = change_column_name(stock_prices, LIST_OF_POSSIBLE_STOCK_COL_NAMES, CORRECT_STOCK_COL_NAME )
    # earnings_dates = change_column_name(earnings_dates, LIST_OF_POSSIBLE_STOCK_COL_NAMES, CORRECT_STOCK_COL_NAME )
    
    # # Sort to prep for merge_asof
    # stock_prices["date"] = parse_date(stock_prices["date"])
    # earnings_dates["earnings_date"] = parse_date(earnings_dates["earnings_date"])

    # stock_prices = stock_prices.sort_values("date")
    # earnings_dates = earnings_dates.sort_values("earnings_date")
    
    # df = merge_prices_earnings_dates(stock_prices, earnings_dates) # df that holds stock prices, earnings dates, EPS data merged

    # sector_market_cap_beta_df = fetch_sectors_market_cap_beta()
    
    # df = map_sector_data_to_main_df(df, sector_market_cap_beta_df)

    # # Sort, make sure "price" is numeric, make sure dates are datetime just in case
    # df = df.sort_values(["stock", "date"]).reset_index(drop=True)
    # df["date"] = parse_date(df["date"])
    # df["earnings_date"] = parse_date(df["earnings_date"])
    # df["price"] = parse_numeric(df["price"])
    
    #return df