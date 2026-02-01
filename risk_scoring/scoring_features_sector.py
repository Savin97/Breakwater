#risk_scoring/scoring_features_sector.py
import pandas as pd

def engineer_sector_vol_stress(input_df: pd.DataFrame, q_high = 0.9) -> pd.DataFrame:
    """

    """
    df = input_df.copy()

    df["sector_vol_ratio_pct"] = (
        df.groupby(["sector", "date"])["vol_ratio_10_to_30"]
            .rank(pct=True, method="average")
    )

    df["sector_vol_stress_high"] = (df["sector_vol_ratio_pct"] >= q_high).astype(int)

    return df