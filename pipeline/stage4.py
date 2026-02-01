# pipeline/stage4.py

cols_to_drop_for_output = [
        # identifiers / bookkeeping
        "fiscal_date_ending",

        # raw price / returns
        "daily_ret",

        # post-event outcome labels (leakage)
        "reaction_1d",
        "reaction_5d",
        "is_up",
        "is_down",
        "is_nochange",

        # intermediate reaction statistics
        "reaction_std",
        "reaction_entropy",
        "directional_bias",
        "abs_reaction_median",
        "abs_reaction_p75",

        # raw features replaced by scores
        "drift_30d",
        "drift_60d",
        "mom_5d",
        "mom_20d",
        "vol_10d",
        "vol_30d",
        "vol_ratio_10_to_30",
        "vol_ratio_cross_percentile",

        # sector internals already abstracted
        "sector_drift_60d",
        "sector_vol_10d",
        "sector_vol_30d",
        "sector_vol_ratio_pct",
    ]