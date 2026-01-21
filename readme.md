Breakwater
An earnings risk model.
First, ingest historical stock prices (Currently 2025-), Earnings dates, EPS data, Sector & Sub-sector data, Market cap, Beta.

Second, Engineer Features:
    Return_1d,3d,5d, daily pct_change:
    ret_1d,ret_3d,ret_5d - price_t+1d,3d,5d - price / price
    daily return - df.groupby('stock')['price'].pct_change()

    Rolling stats:
    3d Drift = df.groupby('stock')['daily'].transform(lambda x: x.rolling(10).mean().shift(1))
    3d Volatility = df.groupby('stock')['daily'].transform(lambda x:x.rolling(10).std().shift(1))
    3d Momentum = df.groupby('stock')['daily'].transform(lambda x: x.rolling(3).sum().shift(1)) 

    Sector Features (Take Averages):
    3d Sector Drift
    3d Sector Volatility

Third, 

Pipeline:
main.py
 -> pipeline.py
    -> pipeline_stage1.py  
        Stage 1 - "Data Ingestion"
        -> Fetches historical stock prices
        -> Fetches earnings dates
        -> Merges them
        -> RETURNS the merged DF


