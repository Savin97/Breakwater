

def classify_large_earnings_move(input_df):
    """
        large_earnings_move = 1 if abs(reaction_3d) ≥ rolling_p75(abs(reaction_3d), window=W)
        window: 20-40 past earnings for that stock; 28
    """     
    df = input_df.copy()
    # Only meaningful on earnings rows
    earnings_df = df["is_earnings_day"] == True

    # Event label
    df["large_earnings_move"] = (
        (df["abs_reaction_3d"] >= df["abs_reaction_p75_rolling"])
        & earnings_df
    ).astype("float")

    # Unknown where insufficient history
    df.loc[df["abs_reaction_p75_rolling"].isna(), "large_earnings_move"] = float("nan")

    return df
    