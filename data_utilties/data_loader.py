# data_utilities/data_loader.py
import pandas as pd
from config import (
    STOCK_PRICES_FILE_PATH,
    EARNINGS_DATES_FILE_PATH
)

def load_raw_data():      
    """Loads stock prices and earnings files and returns raw DataFrames."""
    stock_values = pd.read_csv(STOCK_VALUES_FILE_PATH)
    earning_dates = pd.read_csv(EARNING_DATES_FILE_PATH)
    print("Stock values shape:", stock_values.shape)
    print("Earning dates shape:", earning_dates.shape)
    return stock_values, earning_dates