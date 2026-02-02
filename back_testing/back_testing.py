# back_testing/back_testing.py
import numpy as np

def classify_large_earnings_move(input_df):
    """
        large_earnings_move = 1 if abs(reaction_3d) ≥ rolling_p75(abs(reaction_3d), window=W)
        window: 20-40 past earnings for that stock; 28
    """     
    df = input_df.copy()
    # Only meaningful on earnings rows
    earnings_df = df["is_earnings_day"] == True

    # # Event label
    # df["large_earnings_move"] = (
    #     (df["abs_reaction_3d"] >= df["abs_reaction_p75_rolling"])
    #     & earnings_df
    # ).astype("float")

    conditions = [ 
        df["abs_reaction_3d"] < df["abs_reaction_p75_rolling"],
        ( df["abs_reaction_3d"] >= df["abs_reaction_p75_rolling"] ) &
        ( df["abs_reaction_3d"] < df["abs_reaction_p90_rolling"] ),
        df["abs_reaction_3d"] > df["abs_reaction_p90_rolling"] 
        ]
    
    earnings_move_buckets = [0,1,2] # 0: normal,1: large,2: extreme
        
    
    df["earnings_move_bucket"] = np.select(conditions, earnings_move_buckets)
    
    #     0 = normal
    #     1 = large (p75-p90)
    #     2 = extreme (p90+)
    # )

    # Unknown where insufficient history
    df.loc[df["abs_reaction_p75_rolling"].isna(), "earnings_move_bucket"] = float("nan")

    return df
    