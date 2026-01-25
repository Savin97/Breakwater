import datetime as dt

# Switches
USE_CACHED_DATA_FLAG = True

# Paths
STOCK_NAMES_FILE_PATH = "data_ingestion/stocks.csv"
PRICES_PATH = "data/stock_prices.csv"
EARNINGS_PATH = "data/earnings_dates.csv"
EPS_PATH = "data/eps_data.csv"
SECTORS_PATH = "data/sector_data.csv"
OUTPUT_PATH = "output/"
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# Global Parameters

DEFAULT_START_DATE = dt.date(2008, 1, 1)
STOCKS_START_DATE = "2025-01-01"
STOCKS_END_DATE = "2025-12-31"
REACTION_THRESHOLD = 0.01
SHORT_TERM_DRIFT = 30 # 30 past days
LONG_TERM_DRIFT = 60 # 60 past days
SHORT_TERM_VOLATILITY = 10 # 10 past days
LONG_TERM_VOLATILITY = 30 # 30 past days
SHORT_TERM_MOMENTUM = 5 # 5 past days
LONG_TERM_MOMENTUM = 20 # 20 past days


#   - API Parameters
FREE_ALPHAVANTAGE_KEY = "KMMS5F3XHGRFA7CD"
TIMEOUT_SECONDS = 20.0
BACKOFF_SECONDS = 15.0
MAX_RETRIES = 5
DEFAULT_FETCH_CHUNK_SIZE = 50
CORRECT_STOCK_COL_NAME = "stock"
LIST_OF_POSSIBLE_STOCK_COL_NAMES = ["ticker", "Ticker", "Symbol", "symbol", "Stock", "stock"]
