feature definition explanation
stock
date
earnings_date


Stock Drift |  | Capital flow / macro / theme pressure affecting all stocks in that bucket Earnings reactions are amplified when both Stock drift and Sector drift align.

Pre-earninigs features:
Daily return
Drift (drift_60d) | 60 day stock drift, daily rolling 60 mean | AVG of past 60 daily returns (with .shift(1)) ; expectations about this company
Volatility (vol_10,30d) | 10,30 day Volatility | STD of past 10/30 daily returns (with .shift(1))
Momentum (mom_5,20d) | 5,20 day Momentum | Sum of past 5/20 daily returns (with .shift(1))

days_from_last_earnings
Days to next earnings (Calendar days) = (next_earnings_date - date).days 

earnings_proximity = min(
    abs(days_from_last_earnings),
    abs(days_to_next_earnings)
) # very low → stock is “hot” | mid → digestion phase | high → normal regime

is_earnings_week → days_to_earnings ∈ [0,7]
is_earnings_window → [-2,+3]

abs_reaction_median_3d
abs_reaction_p75_3d | The earnings move size that this stock exceeds only in its top 25% of past 
    earnings reactions | 
    Feature meaning
    ---------------
    abs_reaction_p75_3d represents a "large but typical" earnings move size:
    the absolute 3-day post-earnings return that a stock has exceeded in only
    its top 25% of past earnings reactions.
    
    Typical applications include:
    • Identifying stocks with fat-tailed earnings reactions
    • Normalizing current reactions:
        |DEFAULT_REACTION_WINDOW| / abs_reaction_p75_3d
    • Risk bucketing and position sizing
    • Comparing stock-level earnings risk to sector-level behavior


sector_earnings_count_5d - # of distinct stocks in the same sector with earnings within ±N calendar days
sector_earnings_count_10d - high count → crowded tape | low count → idiosyncratic reaction more likely

sector_earnings_density = (
    sector_earnings_count_5d / sector_size
    )  This fixes: Tech vs Utilities size mismatch, makes values comparable cross-sector
sector_reaction_entropy_5d ; High entropy = mixed Up / Down reactions, no clear narrative, lower predictability

Global earnings pressure
    market_earnings_count_5d
    market_earnings_density