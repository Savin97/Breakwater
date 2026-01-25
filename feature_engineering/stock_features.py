# feature_engineering/stock_features.py
import pandas as pd
from config import (SHORT_TERM_DRIFT,
                    LONG_TERM_DRIFT,
                    SHORT_TERM_VOLATILITY,
                    LONG_TERM_VOLATILITY,
                    SHORT_TERM_MOMENTUM,
                    LONG_TERM_MOMENTUM)

def engineer_daily_ret(input_df):
    df = input_df.copy()
    # Daily return
    df['daily_ret'] = df.groupby('stock')['price'].pct_change()

    return df


def engineer_drift(input_df):
    df = input_df.copy()
    group = df.groupby('stock')['daily_ret']

    df['drift_30d'] = group.transform(lambda x: x.rolling(SHORT_TERM_DRIFT).mean().shift(1))
    df['drift_60d'] = group.transform(lambda x: x.rolling(LONG_TERM_DRIFT).mean().shift(1))

    return df

def engineer_volatility(input_df):
    df = input_df.copy()
    group = df.groupby('stock')['daily_ret']
    # Volatility (short + baseline)
    df['vol_10d'] = group.transform(lambda x: x.rolling(SHORT_TERM_VOLATILITY).std().shift(1))
    df['vol_30d'] = group.transform(lambda x: x.rolling(LONG_TERM_VOLATILITY).std().shift(1))
    # Vol expansion signal
    df['vol_ratio_10_to_30'] = df['vol_10d'] / df['vol_30d']

    return df

def engineer_momentum(input_df):
    df = input_df.copy()
    group = df.groupby('stock')['daily_ret']
    # Momentum (fast + standard)
    df['mom_5d']  = group.transform(lambda x: x.rolling(SHORT_TERM_MOMENTUM).sum().shift(1))
    df['mom_20d'] = group.transform(lambda x: x.rolling(LONG_TERM_MOMENTUM).sum().shift(1))

    return df

def engineer_abs_reaction_median(input_df):
    df = input_df.copy()
    earnings_df = (
    df.loc[df["earnings_date"].notna(), 
            ["stock", "earnings_date", "reaction_3d"]]
        .sort_values(["stock", "earnings_date"])
    )
    earnings_df["abs_reaction_median_3d"] = (
        df.groupby("stock")["reaction_3d"]
        .apply(lambda col: col.abs().expanding().shift(1)) 
    )
