# back_testing/back_testing.py
import numpy as np

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


    