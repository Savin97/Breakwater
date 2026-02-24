# pipeline/backtesting_stage.py
"""    
    Backtesting Stage
    Produces credibility tables (calibration, lift, hit rates, bucket stats, stability by year/sector).
"""
from backtesting.backtesting import backtesting_suite
from backtesting.backtesting_features import (engineer_abs_reaction_3d,
                                              engineer_abs_reaction_p75_rolling,
                                              engineer_abs_reaction_p90_rolling,
                                              classify_large_earnings_move_bucket)
from backtesting.testing_functions import (check_explosiveness_feature, 
                                           three_way_regime_test,
                                           conditional_hit_rate_analysis,
                                           volatility_only_regime_test,
                                           breakwater_regime_test)

def backtesting_stage(input_df):
    df = input_df.copy()
    features = [engineer_abs_reaction_3d,
                engineer_abs_reaction_p75_rolling,
                engineer_abs_reaction_p90_rolling,
                classify_large_earnings_move_bucket]
    for f in features:
        df = f(df)

    # print("Volatility Only:\n")
    # print(volatility_only_regime_test(df))
    # print("\nBreakwater Regime:\n")
    # print(breakwater_regime_test(df))
    # print("\nBreakwater Regime:\n")    
    # backtesting_suite(df)
    # print("\nThree way regime test:\n")
    # print(three_way_regime_test(df))
    """ 
        You now have your first defensible rule:
        Tail risk increases materially when:
        timing_danger is high
        stock_vs_sector_vol ≥ 1
    """
    import numpy as np
    import pandas as pd

    def forward_eval_onefactor(
        df,
        feature_col,
        train_years=range(2005, 2011),
        test_years=range(2011, 2026),
        q=0.90,
        label_col="is_extreme_reaction",
        earn_col="is_earnings_day",
        date_col="date",
    ):
        earn = df[df[earn_col] == 1].dropna(subset=[date_col, label_col, feature_col]).copy()
        earn["year"] = pd.to_datetime(earn[date_col]).dt.year
        earn["y"] = earn[label_col].astype(int)

        train = earn[earn["year"].isin(train_years)].copy()
        test  = earn[earn["year"].isin(test_years)].copy()

        thr = float(train[feature_col].quantile(q))

        def add_regime(sub):
            out = sub.copy()
            out["is_regime"] = out[feature_col] >= thr
            return out

        def stats(sub, split):
            rows = []
            for y, g in add_regime(sub).groupby("year"):
                N = len(g)
                K = int(g["y"].sum())
                base = K / N if N else np.nan

                r = g[g["is_regime"]]
                n = len(r)
                k = int(r["y"].sum())
                reg = k / n if n else np.nan
                lift = (reg / base) if (base and np.isfinite(reg)) else np.nan

                rows.append({
                    "split": split,
                    "year": int(y),
                    "N_earnings": int(N),
                    "baseline_extreme_rate": base,
                    "n_regime": int(n),
                    "regime_extreme_rate": reg,
                    "lift": lift,
                    "regime_share_of_events": (n / N) if N else np.nan,
                    "regime_capture_of_extremes": (k / K) if K else np.nan,
                })
            return pd.DataFrame(rows).sort_values("year")

        return pd.concat([stats(train, "TRAIN"), stats(test, "TEST")], ignore_index=True), thr


    def forward_eval_twofactor_and(
        df,
        feat1, feat2,
        train_years=range(2005, 2011),
        test_years=range(2011, 2026),
        q=0.90,
        label_col="is_extreme_reaction",
        earn_col="is_earnings_day",
        date_col="date",
    ):
        earn = df[df[earn_col] == 1].dropna(subset=[date_col, label_col, feat1, feat2]).copy()
        earn["year"] = pd.to_datetime(earn[date_col]).dt.year
        earn["y"] = earn[label_col].astype(int)

        train = earn[earn["year"].isin(train_years)].copy()
        test  = earn[earn["year"].isin(test_years)].copy()

        thr1 = float(train[feat1].quantile(q))
        thr2 = float(train[feat2].quantile(q))

        def add_regime(sub):
            out = sub.copy()
            out["is_regime"] = (out[feat1] >= thr1) & (out[feat2] >= thr2)
            return out

        def stats(sub, split):
            rows = []
            for y, g in add_regime(sub).groupby("year"):
                N = len(g)
                K = int(g["y"].sum())
                base = K / N if N else np.nan

                r = g[g["is_regime"]]
                n = len(r)
                k = int(r["y"].sum())
                reg = k / n if n else np.nan
                lift = (reg / base) if (base and np.isfinite(reg)) else np.nan

                rows.append({
                    "split": split,
                    "year": int(y),
                    "N_earnings": int(N),
                    "baseline_extreme_rate": base,
                    "n_regime": int(n),
                    "regime_extreme_rate": reg,
                    "lift": lift,
                    "regime_share_of_events": (n / N) if N else np.nan,
                    "regime_capture_of_extremes": (k / K) if K else np.nan,
                })
            return pd.DataFrame(rows).sort_values("year")

        return pd.concat([stats(train, "TRAIN"), stats(test, "TEST")], ignore_index=True), (thr1, thr2)
    backtesting_df = df.copy()
    # 1) Expanding prior only
    prior_stats, prior_thr = forward_eval_onefactor(backtesting_df, "abs_reaction_p75", q=0.90)
    print("PRIOR thr:", prior_thr)
    print(prior_stats[prior_stats["split"]=="TEST"][["year","n_regime","regime_extreme_rate","lift","regime_capture_of_extremes"]].to_string(index=False))

    # 2) Rolling prior only (if you have it)
    roll_stats, roll_thr = forward_eval_onefactor(backtesting_df, "abs_reaction_p75_rolling", q=0.90)
    print("ROLL thr:", roll_thr)
    print(roll_stats[roll_stats["split"]=="TEST"][["year","n_regime","regime_extreme_rate","lift","regime_capture_of_extremes"]].to_string(index=False))

    # 3) Prior + fragility (your current core)
    pf_stats, (p_thr, f_thr) = forward_eval_twofactor_and(backtesting_df, "abs_reaction_p75", "momentum_fragility_score", q=0.90)
    print("PRIOR+FRAG thrs:", p_thr, f_thr)
    print(pf_stats[pf_stats["split"]=="TEST"][["year","n_regime","regime_extreme_rate","lift","regime_capture_of_extremes"]].to_string(index=False))
        
