import pandas as pd
import requests 
URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def fetch_sp500_tickers():
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
    sp500 = tables[0]
    sp500 = sp500.rename(columns={
        "Symbol": "stock",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_sector"
    })
    sp500["stock"] = sp500["stock"].str.replace(".", "-", regex=False)
    # Fetch symbols only
    sp500[["stock"]].to_csv("sp500_symbols.csv", index=False)
    # Fetch symbols, names and setcor data
    #sp500[["stock", "name", "sector", "sub_sector"]].to_csv("sp500_full_info.csv",index=False)


