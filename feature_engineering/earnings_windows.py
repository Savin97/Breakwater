# feature_engineering/earnings_windows.py
import pandas as pd

def engineer_earnings_windows(input_df):
    df = input_df.copy()
    df["days_to_earnings"] = (df["earnings_date"] - df["date"]).dt.days
    df["is_earnings_day"] = ( df["days_to_earnings"].notna()  # Avoids errors when days_to_earnings is N/A which leads to False->0
                               & (df["days_to_earnings"] == 0 ) ).astype("Int64")
    df["is_earnings_week"] = ( df["days_to_earnings"].notna()  # Avoids errors when days_to_earnings is N/A which leads to False->0
                               & (df["days_to_earnings"].between(0, 5)) ).astype("Int64")
    df["is_earnings_window"] = ( df["days_to_earnings"].notna()  # Avoids errors when days_to_earnings is N/A which leads to False->0
                               & (df["days_to_earnings"].between(0, 10)) ).astype("Int64")
    
    return df