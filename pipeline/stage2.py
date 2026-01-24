# stage2.py
""" Pipeline Stage 2
    Input is a df with columns
    stock  date  price_adj_close  earnings_date  fiscal_date_ending reported_eps  estimated_eps  surprise_percentage

    Feature Engineering - daily, ret_1d,ret_3d,ret_5d
"""
import pandas as pd
import numpy as np

from config import REACTION_THRESHOLD, OUTPUT_PATH
from feature_engineering.returns import engineer_returns, classify_reaction

def stage2(stage1_df):
    df = stage1_df.copy()

    # Sort, make sure "price" is numeric just in case
    df = df.sort_values(["stock", "date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Daily return
    df['daily'] = df.groupby('stock')['price'].pct_change()

    group = df.groupby('stock')['daily']

    # Drift
    df['drift_60d'] = group.transform(lambda x: x.rolling(60).mean().shift(1))

    # Volatility (short + baseline)
    df['vol_10d'] = group.transform(lambda x: x.rolling(10).std().shift(1))
    df['vol_30d'] = group.transform(lambda x: x.rolling(30).std().shift(1))

    # Vol expansion signal
    df['vol_ratio_10_to_30'] = df['vol_10d'] / df['vol_30d']

    # Momentum (fast + standard)
    df['mom_5d']  = group.transform(lambda x: x.rolling(5).sum().shift(1))
    df['mom_20d'] = group.transform(lambda x: x.rolling(20).sum().shift(1))

    sector_drift_and_vol = (
        df.groupby(['sector', 'date'])
            .agg(
                sector_drift_60d=('drift_60d', 'mean'), # Average of past 60 days across every stock in the same sector on that day
                sector_vol_10d=('vol_10d', 'mean'), # Typical daily volatility in the sector in the last 10 days, percentages
                sector_vol_30d=('vol_30d', 'mean')
            )
            .groupby('sector')
            .shift(1)
            .reset_index()
    )
    df = df.merge( sector_drift_and_vol, on=['sector', 'date'], how='left')

    df.to_csv(f"{OUTPUT_PATH}/daily_df.csv" ,index = False)

    # DF separation - earnings dates only from here on out
    earnings_df = df[df["date"]==df["earnings_date"]]
    earnings_df.to_csv("E_df.csv")

    earnings_df = engineer_returns(earnings_df)

    earnings_df["ret_1d_class"] = classify_reaction(earnings_df["ret_1d"], REACTION_THRESHOLD)
    earnings_df["ret_3d_class"] = classify_reaction(earnings_df["ret_3d"], REACTION_THRESHOLD)
    earnings_df["ret_5d_class"] = classify_reaction(earnings_df["ret_5d"], REACTION_THRESHOLD)

    return df