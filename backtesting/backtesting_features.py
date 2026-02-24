# back_testing/back_testing_features.py
import numpy as np

def engineer_abs_reaction_3d(df):
    earnings_mask = df["is_earnings_day"] == True
    # Absolute reaction
    df.loc[earnings_mask, "abs_reaction_3d"] = (
        df.loc[earnings_mask, "reaction_3d"].abs()
    )
    return df

def engineer_abs_reaction_p75_rolling(df, window=28, percentile=0.75):
    earnings_df = df["is_earnings_day"] == True
    # Rolling percentile per stock, using past earnings only
    df.loc[earnings_df, "abs_reaction_p75_rolling"] = (
        df.loc[earnings_df]
          .groupby("stock")["abs_reaction_3d"]
          .transform(
              lambda x: (
                  x.shift(1)
                   .rolling(window, min_periods=window)
                   .quantile(percentile)
              )
          )
    )

    return df

def engineer_abs_reaction_p90_rolling(df, window=28, percentile=0.9):
    earnings_df = df["is_earnings_day"] == True
    # Rolling percentile per stock, using past earnings only
    df.loc[earnings_df, "abs_reaction_p90_rolling"] = (
        df.loc[earnings_df]
          .groupby("stock")["abs_reaction_3d"]
          .transform(
              lambda x: (
                  x.shift(1)
                   .rolling(window, min_periods=window)
                   .quantile(percentile)
              )
          )
    )

    return df

def classify_large_earnings_move_bucket(input_df):
    """
        large_earnings_move = 1 if abs_reaction_3d ≥ abs_reaction_p75_rolling
        window: 20-40 past earnings for that stock; 28
    """
    df = input_df.copy()
    # Only meaningful on earnings rows and when p75, p90 aren't NaN
    eligible = (
        df["is_earnings_day"]
        & df[["abs_reaction_p75_rolling", "abs_reaction_p90_rolling"]].notna().all(axis=1)
    )
    conditions = [
        eligible & (df["abs_reaction_3d"] <  df["abs_reaction_p75_rolling"]),
        eligible & (df["abs_reaction_3d"] >= df["abs_reaction_p75_rolling"])
                 & (df["abs_reaction_3d"] <  df["abs_reaction_p90_rolling"]),
        eligible & (df["abs_reaction_3d"] >= df["abs_reaction_p90_rolling"]),
    ]

    # 0 = normal    # 1 = large (p75-p90)    # 2 = extreme (p90+)
    # Unknown where insufficient history
    df["earnings_move_bucket"] = np.select(conditions, [0, 1, 2], default=np.nan) 

    return df