import pandas as pd
import numpy as np

def engineer_returns(input_df):
    df = input_df.copy()

    df = df.sort_values(["stock", "date"])

    # TODO: 24/1/26 Fix this, should be both post rearnings returns, and should also be 3 days before earnings
    # g = df.groupby("stock")["price"]
    # df["ret_1d"] = g.div(g.shift(1)).sub(1)
    # df["ret_3d"] = g.div(g.shift(3)).sub(1)
    # df["ret_5d"] = g.div(g.shift(5)).sub(1)
    return df

def classify_reaction(series : pd.Series, threshold : float) -> np.ndarray:
    """
        Takes in the return column (ret_1d,3d,etc) 

        Returns a column of a classification of the return to
        1 being "Up", -1 being "Down" or 0 "No Change"
    """
    
    return np.select(
        [series > threshold, series < -threshold],
        [1, -1],
        default=0
    )
