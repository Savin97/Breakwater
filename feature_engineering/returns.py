import pandas as pd

def engineer_returns(input_df):
    df = input_df.copy()
    df["ret_1d"] = (
            ( df.groupby("stock")["price"] / 
            df.groupby("stock")["price"].shift(1) ) - 1
        )
        
    df["ret_3d"] = (
        ( df.groupby("stock")["price"] / 
        df.groupby("stock")["price"].shift(1) ) - 1
    )

    df["ret_5d"] = (
        ( df.groupby("stock")["price"] / 
        df.groupby("stock")["price"].shift(1) ) - 1
    )
    return df
