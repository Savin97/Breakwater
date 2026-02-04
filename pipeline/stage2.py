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

from feature_engineering.pre_earnings_sector_features import (engineer_sector_drift_vol,
                                                              engineer_stock_vs_sector_vol,
                                                              engineer_sector_earnings_density)


def stage2(stage1_df):
    """ 
        Pipeline Stage 2
        Input is a df with columns
        stock  date  price  earnings_date  fiscal_date_ending reported_eps  estimated_eps  surprise_percentage

        Pre-Earnings Feature Engineering
        
    """
    stage2_df = stage1_df.copy()
    stage2_df = stage2_df.sort_values(["stock","date"], kind="mergesort")
    stage2_df["date"] = parse_date(stage2_df["date"])
    
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
        engineer_abs_reaction_p75,
        engineer_sector_drift_vol,
        engineer_stock_vs_sector_vol,
        engineer_sector_earnings_density
    ]
    for feature in feature_steps:
        stage2_df = feature(stage2_df)

    if stage2_df is None:
        raise ValueError("\n---ERROR! Stage 2 Returned None.---\n")
    return stage2_df