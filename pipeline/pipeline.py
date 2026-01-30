# pipeline/pipeline.py

from pipeline.stage1 import stage1
from pipeline.stage2 import stage2
from pipeline.stage3 import stage3

def run_pipeline():
    """
        The engine pipeline stages:
        1. Fetch stock prices, earnings data, EPS data.
        2. Merge into a single DataFrame
        3. Engineer features
    """
    
    inputs_df = stage1() 
    feature_engineering = stage2(inputs_df)
    scoring_df = stage3(feature_engineering)

    inputs_df.to_csv("output/stage_1_df.csv", index=False)
    print("\n----------------")
    print("Stage 1 DF created in: output/stage_1_df.csv")

    feature_engineering.to_csv("output/stage_2_df.csv", index=False)
    print("----------------")
    print("Stage 2 DF created in: output/stage_2_df.csv")

    scoring_df.to_csv("output/stage_3_df.csv", index=False)
    print("----------------")
    print("Stage 3 DF created in: output/stage_3_df.csv")
    cols_to_drop = [
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
    output_df = scoring_df.drop(columns=cols_to_drop)
    output_df.to_csv("output/output_df.csv", index=False)
    print("----------------")
    print("output_df created in: output/output_df.csv")

    print("----------------")
    return scoring_df