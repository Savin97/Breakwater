import pandas as pd
import numpy as np

def engineer_earnings_reactions(df):
    """
        Compute forward post-earnings price reactions.

        For each stock and date, computes forward returns:
            reaction_k = price(t + k) / price(t) - 1
        for k in {1, 3, 5} trading days.

        Reactions are computed mechanically for all rows to preserve
        group alignment, then set to NaN on non-earnings days.

        Contract:
        - Requires columns: ["stock", "date", "price", "is_earnings_day"]
        - Output columns exist only on earnings days; non-earnings rows are NaN
        - Prevents leakage of post-event information into normal days
    """
    df = df.copy().sort_values(["stock", "date"])
    group = df.groupby("stock")["price"]

    # forward returns from *today* to +k trading days
    df["reaction_1d"] = (group.shift(-1) / df["price"]) - (1)
    df["reaction_3d"] = (group.shift(-3) / df["price"]) - (1)
    df["reaction_5d"] = (group.shift(-5) / df["price"]) - (1)

    # keep only on earnings days (else NaN)
    mask = df["is_earnings_day"].astype(bool)
    for column in ["reaction_1d", "reaction_3d", "reaction_5d"]:
        df.loc[~mask, column] = np.nan # Apply NaN where the mask returns False

    # Assertion checks
    for i in [1,3,5]:
        assert df.loc[mask, f"reaction_{i}d"].notna().any() # At least 1 has a valid reaction
        assert df.loc[~mask, f"reaction_{i}d"].isna().all() # No reactions on non-earnings days
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
