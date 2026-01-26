# feature_engineering/pre_earnings_stock_features.py
import pandas as pd
import numpy as np

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

def engineer_earnings_windows(input_df):
    """
        Builds features:
        days_to_earnings: Earnings date - current date
        is_earnings_day: Earnings date - current date = 0
        is_earnings_week: Earnings date - current date <= 5
        is_earnings_window: Earnings date - current date <= 10
    """
    df = input_df.copy()
    df["days_to_earnings"] = (df["earnings_date"] - df["date"]).dt.days
    df["is_earnings_day"] = ( df["days_to_earnings"].notna()  # Avoids errors when days_to_earnings is N/A which leads to False->0
                               & (df["days_to_earnings"] == 0 ) ).astype("Int64")
    df["is_earnings_week"] = ( df["days_to_earnings"].notna()  # Avoids errors when days_to_earnings is N/A which leads to False->0
                               & (df["days_to_earnings"].between(0, 5)) ).astype("Int64")
    df["is_earnings_window"] = ( df["days_to_earnings"].notna()  # Avoids errors when days_to_earnings is N/A which leads to False->0
                               & (df["days_to_earnings"].between(0, 10)) ).astype("Int64")
    
    return df

def engineer_abs_reaction_median_3d(input_df):
    """
        Median of |reaction_3d| over past earnings.

        Captures Typical size of earnings moves
        Robust to outliers

        High → this stock usually moves on earnings
        Low → earnings are often a non-event
        Intuition:
            - Median answers: "What usually happens on earnings?"

        Median so one crazy quarter doesn't dominate the signal
    """
    df = input_df.copy()
    # Separate earnings rows
    earnings_mask =  df["reaction_3d"].notna()
    earnings_df = df.loc[earnings_mask, ["stock","earnings_date","reaction_3d"]].copy()
    earnings_df = earnings_df.sort_values(["stock", "earnings_date"])

    # write back only on earnings rows
    earnings_df["abs_reaction_median_3d"] = (
        earnings_df.groupby("stock")["reaction_3d"]
        .transform(lambda x:x.abs().shift(1).expanding().median() ) 
        )
    
    # TODO: .to_numpy might be dangerous. It assumes positional alignment, not logical alignment.
    df.loc[earnings_mask, "abs_reaction_median_3d"] = earnings_df["abs_reaction_median_3d"].to_numpy()
    
    return df


def engineer_abs_reaction_p75_3d(input_df):
    """
        Compute the 75th percentile of historical absolute 3-day earnings reactions
        for each stock, using only *past* earnings events.

        Intuition:
            - p75 answers: "When it moves meaningfully, how big does it often get?"
        This captures the stock's *upper-tail earnings volatility* without being
        dominated by single extreme outliers (unlike max or std).

        Construction details
        --------------------
        • Strictly non-leaky: statistics at time t are computed from events < t
        via shift(1)
        • Expanding window over the stock's earnings history
        • First earnings event per stock has no past history → NaN (by design)
        Values are populated only on earnings rows; non-earnings rows remain NaN.
        
    """
    df = input_df.copy()
    # Separate earnings rows
    earnings_mask =  df["reaction_3d"].notna()
    earnings_df = df.loc[earnings_mask, ["stock","earnings_date","reaction_3d"]].copy()
    earnings_df = earnings_df.sort_values(["stock", "earnings_date"])

    # write back only on earnings rows
    earnings_df["abs_reaction_p75_3d"] = (
        earnings_df.groupby("stock")["reaction_3d"]
        .transform(lambda x:x.abs().shift(1).expanding().quantile(0.75) ) 
        )
    
    # TODO: .to_numpy might be dangerous. It assumes positional alignment, not logical alignment.
    df.loc[earnings_mask, "abs_reaction_p75_3d"] = earnings_df["abs_reaction_p75_3d"].to_numpy()

    return df

