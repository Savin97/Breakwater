import pandas as pd

def engineer_sector_drift_vol(df):
        
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
        return df