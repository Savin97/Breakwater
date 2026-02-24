# pipeline/stage2.py
import duckdb
from config import (CORRECT_STOCK_COL_NAME,
                    LIST_OF_POSSIBLE_STOCK_COL_NAMES,
                    PRICES_PROVIDER,
                    DB_PATH)
from data_utilities.formatting import parse_date, parse_numeric, change_column_name
from data_ingestion.fetch_stock_prices import fetch_stock_prices
from data_ingestion.fetch_earnings import fetch_earnings_dates
from data_ingestion.fetch_eps import fetch_eps
from data_ingestion.fetch_sectors import fetch_sectors_market_cap_beta
from data_utilities.merging import (merge_prices_earnings_dates, 
                                    merge_main_df_with_eps_df, 
                                    map_sector_data_to_main_df)
from data_utilities.helper_funcs import read_stocks_to_fetch
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
    print("Stage 2...")
    con = duckdb.connect(DB_PATH)
    prices_df = con.execute("SELECT stock,price,date FROM prices ORDER BY stock,date").fetch_df()
    earnings_df = con.execute("SELECT stock,earnings_date FROM earnings ORDER BY stock,earnings_date").fetch_df()
    # stock_data_df = con.execute("SELECT * FROM stock_data ORDER BY stock").fetch_df()
    
    prices_df["date"] = parse_date(prices_df["date"])
    earnings_df["earnings_date"] = parse_date(earnings_df["earnings_date"])

    prices_df = prices_df.sort_values("date")
    earnings_df = earnings_df.sort_values("earnings_date")
    
    df = merge_prices_earnings_dates(prices_df, earnings_df) # df that holds stock prices, earnings dates, EPS data merged

    sector_market_cap_beta_df = fetch_sectors_market_cap_beta()
    
    df = map_sector_data_to_main_df(df, sector_market_cap_beta_df)

    # Sort, make sure "price" is numeric, make sure dates are datetime just in case
    df = df.sort_values(["stock", "date"]).reset_index(drop=True)
    df["date"] = parse_date(df["date"])
    df["earnings_date"] = parse_date(df["earnings_date"])
    df["price"] = parse_numeric(df["price"])
    # print(df.head())
    # print("\n\n\n")
    # print(df.tail())
    # print("\n\n\n")
    # print(df.columns)
    return df


    # importing sector data
    # import yfinance as yf, pandas as pd, requests
    # sector_df = pd.DataFrame()
    # sector_list = []

    # URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # #for i,stock in enumerate(stock_list,start=1):
    # #print(f"({i}/{len(stock_list)}) Fetching {stock} sector info...")
    # headers = {
    # "User-Agent": (
    #         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    #         "AppleWebKit/537.36 (KHTML, like Gecko) "
    #         "Chrome/120.0.0.0 Safari/537.36"
    #     )
    # }
    # exit()
    # r = requests.get(URL, headers=headers, timeout=30)
    # r.raise_for_status()  # will show you 403/429 clearly if it still happens

    # tables = pd.read_html(r.text)
    # sp500 = tables[0]
    # sp500_changes = tables[1] # TODO: might be useful later for organizing changes in the stock list
    # sp500 = sp500.rename(columns={
    #     "Symbol": "stock",
    #     "Security": "name",
    #     "GICS Sector": "sector",
    #     "GICS Sub-Industry": "sub_sector"
    # })
    # #print(sp500)
    # #print(sp500[sp500["stock"]=="MMM"])
    # #print(sp500["stock"].dtype)
    # #print(sp500["stock"].values)
    # sp500.sort_values("stock")
    # sp500["stock"] = sp500["stock"].str.replace(".", "-", regex=False)
    # # Fetch symbols only
    # for stock in stock_list:
    #     if stock not in sp500["stock"].values:
    #         print(f"{stock} Not in table")

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