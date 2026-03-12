# data_ingestion/fetch_sp500_sectors.py
from data_ingestion.data_utilities import read_stocks_to_fetch
import pandas as pd, requests

def get_sp500_sectors():
    # importing stock list and sector data
    URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    stock_list = read_stocks_to_fetch()
    headers = {
    "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    r = requests.get(URL, headers=headers, timeout=30)
    r.raise_for_status()  # will show you 403/429 clearly if it still happens
    tables = pd.read_html(r.text)
    sp500_df = tables[0]
    sp500_changes = tables[1] # TODO: might be useful later for organizing changes in the stock list
    sp500_df = sp500_df.rename(columns={
        "Symbol": "stock",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_sector"
    })
    sp500_df = sp500_df[["stock","name","sector","sub_sector"]]
    sp500_df.sort_values("stock")
    sp500_df["stock"] = sp500_df["stock"].str.replace(".", "-", regex=False)
    # Fetch symbols only
    for stock in stock_list:
        if stock not in sp500_df["stock"].values:
            print(f"PROBLEM! {stock} not in my stock_list file")
    stock_list_df = sp500_df["stock"]
    return sp500_df

def ingest_all_sector_data():
    pass
