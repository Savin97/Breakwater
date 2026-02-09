# back_testing/back_testing_features.py
def engineer_abs_reaction_3d(df):
    earnings_df = df["is_earnings_day"] == True
    # Absolute reaction
    df.loc[earnings_df, "abs_reaction_3d"] = (
        df.loc[earnings_df, "reaction_3d"].abs()
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