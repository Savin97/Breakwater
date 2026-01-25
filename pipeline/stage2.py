# stage2.py
import pandas as pd
import numpy as np

from config import (REACTION_THRESHOLD, 
                    OUTPUT_PATH )
from feature_engineering.stock_features import (engineer_daily_ret,
                                                engineer_drift, 
                                                engineer_volatility, 
                                                engineer_momentum,
                                                engineer_abs_reaction_median, 
                                                engineer_abs_reaction_p75_3d)
from feature_engineering.returns import engineer_earnings_reactions, classify_reaction
from feature_engineering.earnings_windows import engineer_earnings_windows

def stage2(stage1_df):
    """ 
        Pipeline Stage 2
        Input is a df with columns
        stock  date  price  earnings_date  fiscal_date_ending reported_eps  estimated_eps  surprise_percentage

        Pre-Earnings Feature Engineering
        
    """
    df = stage1_df.copy()

    feature_steps = [
        engineer_daily_ret,
        engineer_drift,
        engineer_volatility,
        engineer_momentum,
        engineer_earnings_windows,
        engineer_earnings_reactions,
        engineer_abs_reaction_median,
        engineer_abs_reaction_p75_3d
    ]
    for feature in feature_steps:
        df = feature(df)

    

    # Sector Features
    # sector_drift_and_vol = (
    #     df.groupby(['sector', 'date'])
    #         .agg(
    #             sector_drift_60d=('drift_60d', 'mean'), # Average of past 60 days across every stock in the same sector on that day
    #             sector_vol_10d=('vol_10d', 'mean'), # Typical daily volatility in the sector in the last 10 days, percentages
    #             sector_vol_30d=('vol_30d', 'mean')
    #         )
    #         .groupby('sector')
    #         .shift(1)
    #         .reset_index()
    # )
    # df = df.merge( sector_drift_and_vol, on=['sector', 'date'], how='left')

    return df