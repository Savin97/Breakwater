# feature_engineering/pre_earnings_sector_features.py
"""
    Pre-earnings sector features:
    sector_drift_{SHORT_TERM_DRIFT}d # 30 days by default
    sector_drift_{LONG_TERM_DRIFT}d # 60 days by default
    sector_vol_30d
"""
import numpy as np

def engineer_sector_drift_vol(df):     
    sector_drift_and_vol = (
        df.groupby(['sector', 'date'])
            .agg(
                sector_drift_60d=('drift_60d', 'mean'), # Average of past 60 days across every stock in the same sector on that day
                sector_vol_10d=('vol_10d', 'mean'), # Typical daily volatility in the sector in the last 10 days, percentages
                sector_vol_30d=('vol_30d', 'mean')
            )
            .sort_values(["sector", "date"])
        )
    
    # Prevent same-day leakage of sector aggregates (optional-but-sane)
    cols_to_shift = ["sector_drift_60d", "sector_vol_10d", "sector_vol_30d"]
    sector_drift_and_vol[cols_to_shift] = (
        sector_drift_and_vol.groupby("sector")[cols_to_shift].shift(1)
    )

    df = df.merge( sector_drift_and_vol, on=['sector', 'date'], how='left')
    return df

def engineer_stock_vs_sector_vol(df):
    """
        relative volatility: “is this stock calmer or crazier than its sector?”

        Values ~1.0 = sector-typical
        1.5 = 50% more volatile than sector
        0.7 = calmer than sector
    """
    denom = df["sector_vol_30d"].replace(0, np.nan) # prevent divison by zero
    df["stock_vs_sector_vol"] = df["vol_30d"] / denom
    return df

def engineer_sector_earnings_density(input_df):
    """
        Shows fraction of stocks in the sector whose earnings are within the next week.
        Calculated by mean.
    """
    df = input_df.sort_values(["sector", "date"])

    df["sector_earnings_density"] = (df.groupby(["sector","date"])["is_earnings_week"]
            .transform("mean")
        )
    df = df.sort_values(["stock", "date"]).reset_index(drop=True)
    return df


