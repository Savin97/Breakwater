# stage2.py

from data_utilities.formatting import parse_date
from feature_engineering.pre_earnings_stock_features import (engineer_daily_ret,
                                                engineer_drift, 
                                                engineer_volatility, 
                                                engineer_momentum,
                                                engineer_abs_reaction_median, 
                                                engineer_abs_reaction_p75,
                                                engineer_earnings_windows )

from feature_engineering.post_earnings_stock_features import (engineer_earnings_reactions,
                                                              engineer_reaction_class,
                                                              engineer_reaction_std,
                                                              engineer_reaction_entropy,
                                                              engineer_directional_bias)


def stage2(stage1_df):
    """ 
        Pipeline Stage 2
        Input is a df with columns
        stock  date  price  earnings_date  fiscal_date_ending reported_eps  estimated_eps  surprise_percentage

        Pre-Earnings Feature Engineering
        
    """
    df = stage1_df.copy()
    df = df.sort_values(["stock","date"], kind="mergesort")
    df["date"] = parse_date(df["date"])

    feature_steps = [
        engineer_daily_ret,
        engineer_drift,
        engineer_volatility,
        engineer_momentum,
        engineer_earnings_windows,
        engineer_earnings_reactions,
        engineer_reaction_class,
        engineer_reaction_std,
        engineer_reaction_entropy,
        engineer_directional_bias,
        engineer_abs_reaction_median,
        engineer_abs_reaction_p75
    ]
    for feature in feature_steps:
        df = feature(df)

    if df is None:
        raise ValueError("\n---ERROR! Stage 2 Returned None.---\n")
    return df