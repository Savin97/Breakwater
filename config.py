# config.py

# Switches
USE_CACHED_DATA_FLAG = True
USE_CACHED_PRICES_DATA_FLAG = True
USE_CACHED_EARNINGS_DATA_FLAG = True
USE_CACHED_SECTOR_DATA_FLAG = True
USE_CACHED_EPS_DATA_FLAG = True

# Paths
STOCK_NAMES_FILE_PATH = "data_ingestion/stocks.csv"
PRICES_PATH = "data/stock_prices.csv"
EARNINGS_PATH = "data/earnings_dates.csv"
EPS_PATH = "data/eps_data.csv"
SECTORS_PATH = "data/sector_data.csv"
OUTPUT_PATH = "output/"
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# Global Parameters
STOCKS_START_DATE = "2021-01-01"
STOCKS_END_DATE = "2026-01-01"
DEFAULT_REACTION_WINDOW = "reaction_3d" # Model will use 3 days after earnings
REACTION_THRESHOLD = 0.007
SHORT_TERM_DRIFT = 30 # 30 past days
LONG_TERM_DRIFT = 60 # 60 past days
SHORT_TERM_VOLATILITY = 10 # 10 past days
LONG_TERM_VOLATILITY = 30 # 30 past days
SHORT_TERM_MOMENTUM = 5 # 5 past days
LONG_TERM_MOMENTUM = 20 # 20 past days


#   - API Parameters
PRICES_PROVIDER = "ALPHAVANTAGE"
ALPHAVANTAGE_CALLS_PER_MINUTE=75
TIMEOUT_SECONDS = 30.0
BACKOFF_SECONDS = 20.0
MAX_RETRIES = 5
DEFAULT_FETCH_CHUNK_SIZE = 50
CORRECT_STOCK_COL_NAME = "stock"
LIST_OF_POSSIBLE_STOCK_COL_NAMES = ["ticker", "Ticker", "Symbol", "symbol", "Stock", "stock"]



# Parameter for output, change later
cols_to_drop_for_output = [
        # identifiers / bookkeeping
        "fiscal_date_ending",
        # raw price / returns
        "daily_ret",
        # post-event outcome labels (leakage)
        "reaction_1d",
        "reaction_5d",
        "is_up",
        "is_down",
        "is_nochange",
        # intermediate reaction statistics
        "reaction_std",
        "reaction_entropy",
        "directional_bias",
        "abs_reaction_median",
        "abs_reaction_p75",
        # raw features replaced by scores
        "drift_30d",
        "drift_60d",
        "mom_5d",
        "mom_20d",
        "vol_10d",
        "vol_30d",
        "vol_ratio_10_to_30",
        "vol_ratio_cross_percentile",
        # sector internals already abstracted
        "sector_drift_60d",
        "sector_vol_10d",
        "sector_vol_30d",
        "sector_vol_ratio_pct",
    ]
