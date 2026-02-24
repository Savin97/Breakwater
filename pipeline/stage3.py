# pipeline/stage3.py
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

def stage3(stage2_df):
    """ 
        Pipeline Stage 3 - Feature Engineering
        Input is a df with columns
        stock  date  price  earnings_date  fiscal_date_ending reported_eps  estimated_eps  surprise_percentage

        Adds:
        daily_ret = price/yesterday's price
        drift_30d,60d = mean of rolling 30/60 day daily_ret, shift(1)
        vol_10d,30d = STD of rolling 10/30 day daily_ret, shift(1)
        vol_ratio_10_to_30 = vol_10d / vol_30d
        mom_5d,20d = sum of rolling 5/20 day daily_ret, shift(1)
        days_to_earnings = next earnings_date - today's date
        is_earnings_day = 1 if days_to_earnings == 0, else it's 0
        is_earnings_week = 1 if days_to_earnings is between(0, 5), else it's 0
        is_earnings_window = 1 if days_to_earnings is between(0, 10), else it's 0
        reaction_1d,3d,5d = stock price 1d/3d/5d after earnings_date
        is_up/is_down/is_nochange = reaction_3d above, below REACTION_THRESHOLD
        reaction_std = rolling std of |reaction_3d| values, shift(1), min_periods = 3, window = 8 periods
        reaction_entropy = Shannon entropy of the histogram of absolute past reactions, gives how unpredictable and distributionally diverse the stock is
        directional_bias = For each earnings event, expanding mean of past signed reactions for that stock, no leakage.
        abs_reaction_median = Median of |reaction_3d| over past earnings, shift(1)
        abs_reaction_p75 = 75th percentile of historical |reaction_3d| earnings reactions for each stock, using only past earnings events
        sector_drift_60d = mean of drift_60d values for that sector's stocks
        sector_vol_10d,30d = mean of vol_10d/30d values for that sector's stocks
        stock_vs_sector_vol = ratio vol_30d / sector_vol_30d
        sector_earnings_density = fraction of stocks in the sector whose earnings are within the next week, mean of is_earnings_week per sector
    """
    print("Stage 3...")
    stage3_df = stage2_df.copy()
    stage3_df = stage3_df.sort_values(["stock","date"], kind="mergesort")
    stage3_df["date"] = parse_date(stage3_df["date"])
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
        stage3_df = feature(stage3_df)
    # stage3_df.to_csv("stage3_df.csv", index=False)
    # exit()
    if stage3_df is None:
        raise ValueError("\n---ERROR! Stage 2 Returned None.---\n")
    return stage3_df