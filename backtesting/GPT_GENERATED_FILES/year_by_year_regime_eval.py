"""
year_by_year_regime_eval.py

Year-by-year validation for:
  earnings_explosiveness_score (high) + momentum_fragility_score (high)

Outputs, per year:
  - total earnings events (N)
  - total extreme events
  - baseline extreme rate
  - regime size (n)
  - regime extreme events
  - regime extreme rate
  - lift = regime_rate / baseline_rate
  - (optional) Wilson 95% CI for baseline and regime rates

Designed to drop into your Breakwater pipeline as a diagnostic stage.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Stats helpers
# -----------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.
    Returns (low, high). If n==0 returns (nan, nan).
    """
    if n <= 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = (z * math.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def safe_rate(k: int, n: int) -> float:
    return float(k / n) if n > 0 else np.nan

def parse_years(spec: str) -> list[int]:
    years: list[int] = []
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a_i, b_i = int(a), int(b)
            if b_i < a_i:
                raise ValueError(f"Bad range: {p}")
            years.extend(range(a_i, b_i + 1))
        else:
            years.append(int(p))
    return sorted(set(years))


def year_by_year_stats(df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    earnings_df = df[df[cfg.earnings_flag_col] == True].copy()
    earnings_df = earnings_df.dropna(subset=[cfg.date_col, cfg.extreme_label_col, cfg.explosiveness_col, cfg.fragility_col])
    earnings_df["year"] = earnings_df[cfg.date_col].dt.year
    earnings_df["__label__"] = earnings_df[cfg.extreme_label_col].astype(int)

    # global thresholds computed on ALL earnings rows
    exp_thr, frag_thr = compute_global_thresholds(earnings_df, cfg)

    earnings_df["is_joint_regime"] = (
        (earnings_df[cfg.explosiveness_col] >= exp_thr) &
        (earnings_df[cfg.fragility_col] >= frag_thr)
    )

    rows = []
    for y, g in earnings_df.groupby("year"):
        N = int(len(g))
        K = int(g["__label__"].sum())
        base = safe_rate(K, N)

        reg = g[g["is_joint_regime"]]
        n = int(len(reg))
        k = int(reg["__label__"].sum())
        reg_rate = safe_rate(k, n)

        lift = (reg_rate / base) if (np.isfinite(reg_rate) and np.isfinite(base) and base > 0) else np.nan

        rows.append({
            "split": "ALL",
            "year": int(y),
            "N_earnings": N,
            "baseline_extreme_rate": base,
            "n_regime": n,
            "regime_extreme_rate": reg_rate,
            "lift": lift,
            "regime_share_of_events": (n / N) if N > 0 else np.nan,
            "regime_capture_of_extremes": (k / K) if K > 0 else np.nan,
        })

    stats_df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

    # summary row
    N_all = int(len(earnings_df))
    K_all = int(earnings_df["__label__"].sum())
    base_all = safe_rate(K_all, N_all)
    reg_all = earnings_df[earnings_df["is_joint_regime"]]
    n_all = int(len(reg_all))
    k_all = int(reg_all["__label__"].sum())
    reg_rate_all = safe_rate(k_all, n_all)
    lift_all = (reg_rate_all / base_all) if (np.isfinite(reg_rate_all) and np.isfinite(base_all) and base_all > 0) else np.nan

    stats_df = pd.concat([stats_df, pd.DataFrame([{
        "split": "ALL",
        "year": -1,
        "N_earnings": N_all,
        "baseline_extreme_rate": base_all,
        "n_regime": n_all,
        "regime_extreme_rate": reg_rate_all,
        "lift": lift_all,
        "regime_share_of_events": (n_all / N_all) if N_all > 0 else np.nan,
        "regime_capture_of_extremes": (k_all / K_all) if K_all > 0 else np.nan,
    }])], ignore_index=True)

    thresholds_df = pd.DataFrame([{
        "exp_thr": exp_thr,
        "frag_thr": frag_thr,
        "exp_mode": cfg.explosiveness_high_mode,
        "frag_mode": cfg.fragility_high_mode,
        "exp_q": cfg.explosiveness_q,
        "frag_q": cfg.fragility_q,
    }])

    return stats_df, thresholds_df


def forward_validation_stats(
    df: pd.DataFrame,
    cfg,
    train_years_spec: str,
    test_years_spec: str,
    report_train: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    earnings_df = df[df[cfg.earnings_flag_col] == True].copy()
    earnings_df = earnings_df.dropna(subset=[cfg.date_col, cfg.extreme_label_col, cfg.explosiveness_col, cfg.fragility_col])
    earnings_df["year"] = earnings_df[cfg.date_col].dt.year
    earnings_df["__label__"] = earnings_df[cfg.extreme_label_col].astype(int)

    train_years = parse_years(train_years_spec)
    test_years = parse_years(test_years_spec)

    train = earnings_df[earnings_df["year"].isin(train_years)].copy()
    test = earnings_df[earnings_df["year"].isin(test_years)].copy()

    if len(train) == 0:
        raise ValueError("No rows in TRAIN split.")
    if len(test) == 0:
        raise ValueError("No rows in TEST split.")

    # thresholds learned only on TRAIN
    exp_thr, frag_thr = compute_global_thresholds(train, cfg)

    def add_flags(sub: pd.DataFrame) -> pd.DataFrame:
        out = sub.copy()
        out["is_joint_regime"] = (
            (out[cfg.explosiveness_col] >= exp_thr) &
            (out[cfg.fragility_col] >= frag_thr)
        )
        return out

    outs = []
    for label, sub in [("TRAIN", train), ("TEST", test)]:
        if label == "TRAIN" and not report_train:
            continue
        sub = add_flags(sub)
        rows = []
        for y, g in sub.groupby("year"):
            N = int(len(g))
            K = int(g["__label__"].sum())
            base = safe_rate(K, N)

            reg = g[g["is_joint_regime"]]
            n = int(len(reg))
            k = int(reg["__label__"].sum())
            reg_rate = safe_rate(k, n)
            lift = (reg_rate / base) if (np.isfinite(reg_rate) and np.isfinite(base) and base > 0) else np.nan

            rows.append({
                "split": label,
                "year": int(y),
                "N_earnings": N,
                "baseline_extreme_rate": base,
                "n_regime": n,
                "regime_extreme_rate": reg_rate,
                "lift": lift,
                "regime_share_of_events": (n / N) if N > 0 else np.nan,
                "regime_capture_of_extremes": (k / K) if K > 0 else np.nan,
            })
        outs.append(pd.DataFrame(rows).sort_values("year"))

    stats_df = pd.concat(outs, ignore_index=True)

    thresholds_df = pd.DataFrame([{
        "train_years": ",".join(map(str, train_years)),
        "test_years": ",".join(map(str, test_years)),
        "exp_thr": exp_thr,
        "frag_thr": frag_thr,
        "exp_mode": cfg.explosiveness_high_mode,
        "frag_mode": cfg.fragility_high_mode,
        "exp_q": cfg.explosiveness_q,
        "frag_q": cfg.fragility_q,
    }])

    return stats_df, thresholds_df


# -----------------------------
# Config
# -----------------------------
@dataclass
class RegimeConfig:
    # Column names
    date_col: str = "date"
    earnings_flag_col: str = "is_earnings_day"
    extreme_label_col: str = "is_extreme_reaction"  # change if your label differs
    explosiveness_col: str = "earnings_explosiveness_score"
    fragility_col: str = "momentum_fragility_score"

    # How to define "high"
    explosiveness_high_mode: str = "quantile"  # {"quantile","threshold"}
    fragility_high_mode: str = "quantile"      # {"quantile","threshold"}

    explosiveness_q: float = 0.90
    fragility_q: float = 0.90

    explosiveness_threshold: float = 90.0  # if mode="threshold"
    fragility_threshold: float = 80.0      # if mode="threshold"

    # Threshold computation scope for quantiles:
    # "global" => compute once on entire dataset (earnings rows)
    # "year"   => compute separately per year (harder, more adaptive)
    quantile_scope: str = "global"          # {"global","year"}


# -----------------------------
# Core logic
# -----------------------------

def normalize_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce", utc=False)
    return out


def compute_global_thresholds(earnings_df: pd.DataFrame, cfg: RegimeConfig) -> Tuple[float, float]:
    if cfg.explosiveness_high_mode == "quantile":
        exp_thr = float(earnings_df[cfg.explosiveness_col].quantile(cfg.explosiveness_q))
    else:
        exp_thr = float(cfg.explosiveness_threshold)

    if cfg.fragility_high_mode == "quantile":
        frag_thr = float(earnings_df[cfg.fragility_col].quantile(cfg.fragility_q))
    else:
        frag_thr = float(cfg.fragility_threshold)

    return exp_thr, frag_thr


def apply_regime_flags(
    earnings_df: pd.DataFrame, cfg: RegimeConfig, exp_thr: Optional[float] = None, frag_thr: Optional[float] = None
) -> pd.DataFrame:
    out = earnings_df.copy()

    if cfg.explosiveness_high_mode == "threshold" and exp_thr is None:
        exp_thr = cfg.explosiveness_threshold
    if cfg.fragility_high_mode == "threshold" and frag_thr is None:
        frag_thr = cfg.fragility_threshold

    if exp_thr is None or frag_thr is None:
        raise ValueError("exp_thr and frag_thr must be provided for apply_regime_flags().")

    out["is_high_explosiveness"] = out[cfg.explosiveness_col] >= exp_thr
    out["is_high_fragility"] = out[cfg.fragility_col] >= frag_thr
    out["is_joint_regime"] = out["is_high_explosiveness"] & out["is_high_fragility"]
    return out


# def year_by_year_stats(df: pd.DataFrame, cfg: RegimeConfig) -> pd.DataFrame:
#     # Keep only earnings rows
#     earnings_df = df[df[cfg.earnings_flag_col] == True].copy()

#     # Basic cleaning
#     earnings_df = earnings_df.dropna(subset=[cfg.date_col, cfg.extreme_label_col])

#     earnings_df["year"] = earnings_df[cfg.date_col].dt.year

#     # Decide thresholds
#     if cfg.quantile_scope == "global":
#         exp_thr, frag_thr = compute_global_thresholds(earnings_df, cfg)
#         earnings_df = apply_regime_flags(earnings_df, cfg, exp_thr=exp_thr, frag_thr=frag_thr)
#         thresholds_info = pd.DataFrame([{
#             "exp_thr": exp_thr,
#             "frag_thr": frag_thr,
#             "exp_mode": cfg.explosiveness_high_mode,
#             "frag_mode": cfg.fragility_high_mode,
#             "exp_q": cfg.explosiveness_q,
#             "frag_q": cfg.fragility_q,
#             "quantile_scope": cfg.quantile_scope,
#         }])
#     elif cfg.quantile_scope == "year":
#         # Compute per year (more adaptive; can inflate apparent stability a bit)
#         thresholds = (
#             earnings_df.groupby("year")
#             .apply(lambda g: pd.Series({
#                 "exp_thr": float(g[cfg.explosiveness_col].quantile(cfg.explosiveness_q))
#                           if cfg.explosiveness_high_mode == "quantile" else float(cfg.explosiveness_threshold),
#                 "frag_thr": float(g[cfg.fragility_col].quantile(cfg.fragility_q))
#                            if cfg.fragility_high_mode == "quantile" else float(cfg.fragility_threshold),
#             }))
#             .reset_index()
#         )
#         earnings_df = earnings_df.merge(thresholds, on="year", how="left")
#         earnings_df["is_high_explosiveness"] = earnings_df[cfg.explosiveness_col] >= earnings_df["exp_thr"]
#         earnings_df["is_high_fragility"] = earnings_df[cfg.fragility_col] >= earnings_df["frag_thr"]
#         earnings_df["is_joint_regime"] = earnings_df["is_high_explosiveness"] & earnings_df["is_high_fragility"]

#         thresholds_info = thresholds.copy()
#         thresholds_info["exp_mode"] = cfg.explosiveness_high_mode
#         thresholds_info["frag_mode"] = cfg.fragility_high_mode
#         thresholds_info["exp_q"] = cfg.explosiveness_q
#         thresholds_info["frag_q"] = cfg.fragility_q
#         thresholds_info["quantile_scope"] = cfg.quantile_scope
#     else:
#         raise ValueError("quantile_scope must be 'global' or 'year'")

#     # Aggregate stats per year
#     rows = []
#     for y, g in earnings_df.groupby("year"):
#         N = int(len(g))
#         K = int(g[cfg.extreme_label_col].sum())
#         base = safe_rate(K, N)

#         reg = g[g["is_joint_regime"]]
#         n = int(len(reg))
#         k = int(reg[cfg.extreme_label_col].sum())
#         reg_rate = safe_rate(k, n)

#         lift = (reg_rate / base) if (np.isfinite(reg_rate) and np.isfinite(base) and base > 0) else np.nan

#         base_ci = wilson_ci(K, N)
#         reg_ci = wilson_ci(k, n)

#         rows.append({
#             "year": int(y),
#             "N_earnings": N,
#             "K_extreme": K,
#             "baseline_extreme_rate": base,
#             "baseline_ci_low": base_ci[0],
#             "baseline_ci_high": base_ci[1],
#             "n_regime": n,
#             "k_regime_extreme": k,
#             "regime_extreme_rate": reg_rate,
#             "regime_ci_low": reg_ci[0],
#             "regime_ci_high": reg_ci[1],
#             "lift": lift,
#             "regime_share_of_events": (n / N) if N > 0 else np.nan,
#             "regime_capture_of_extremes": (k / K) if K > 0 else np.nan,
#         })

#     out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

#     # Add global summary row at bottom
#     N_all = int(len(earnings_df))
#     K_all = int(earnings_df[cfg.extreme_label_col].sum())
#     base_all = safe_rate(K_all, N_all)

#     reg_all = earnings_df[earnings_df["is_joint_regime"]]
#     n_all = int(len(reg_all))
#     k_all = int(reg_all[cfg.extreme_label_col].sum())
#     reg_rate_all = safe_rate(k_all, n_all)
#     lift_all = (reg_rate_all / base_all) if (np.isfinite(reg_rate_all) and np.isfinite(base_all) and base_all > 0) else np.nan

#     base_ci_all = wilson_ci(K_all, N_all)
#     reg_ci_all = wilson_ci(k_all, n_all)

#     summary = pd.DataFrame([{
#         "year": -1,
#         "N_earnings": N_all,
#         "K_extreme": K_all,
#         "baseline_extreme_rate": base_all,
#         "baseline_ci_low": base_ci_all[0],
#         "baseline_ci_high": base_ci_all[1],
#         "n_regime": n_all,
#         "k_regime_extreme": k_all,
#         "regime_extreme_rate": reg_rate_all,
#         "regime_ci_low": reg_ci_all[0],
#         "regime_ci_high": reg_ci_all[1],
#         "lift": lift_all,
#         "regime_share_of_events": (n_all / N_all) if N_all > 0 else np.nan,
#         "regime_capture_of_extremes": (k_all / K_all) if K_all > 0 else np.nan,
#     }])

#     out = pd.concat([out, summary], ignore_index=True)

#     return out, thresholds_info


def main(df):
    p = argparse.ArgumentParser()
    p.add_argument("--outfile", default="year_by_year_regime_eval.csv", help="CSV output path.")
    p.add_argument("--thresholds_out", default="year_by_year_thresholds.csv", help="Thresholds output path.")
    p.add_argument("--date_col", default="date")
    p.add_argument("--earnings_flag_col", default="is_earnings_day")
    p.add_argument("--extreme_label_col", default="is_extreme_reaction")
    p.add_argument("--explosiveness_col", default="earnings_explosiveness_score")
    p.add_argument("--fragility_col", default="momentum_fragility_score")

    p.add_argument("--exp_mode", choices=["quantile", "threshold"], default="quantile")
    p.add_argument("--frag_mode", choices=["quantile", "threshold"], default="quantile")
    p.add_argument("--exp_q", type=float, default=0.90)
    p.add_argument("--frag_q", type=float, default=0.90)
    p.add_argument("--exp_thr", type=float, default=90.0)
    p.add_argument("--frag_thr", type=float, default=80.0)
    p.add_argument("--quantile_scope", choices=["global", "year"], default="global")

    args = p.parse_args()

    cfg = RegimeConfig(
        date_col=args.date_col,
        earnings_flag_col=args.earnings_flag_col,
        extreme_label_col=args.extreme_label_col,
        explosiveness_col=args.explosiveness_col,
        fragility_col=args.fragility_col,
        explosiveness_high_mode=args.exp_mode,
        fragility_high_mode=args.frag_mode,
        explosiveness_q=args.exp_q,
        fragility_q=args.frag_q,
        explosiveness_threshold=args.exp_thr,
        fragility_threshold=args.frag_thr,
        quantile_scope=args.quantile_scope,
    )

    df = normalize_date(df, cfg.date_col)

    required = [cfg.date_col, cfg.earnings_flag_col, cfg.extreme_label_col, cfg.explosiveness_col, cfg.fragility_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    stats_df, thresholds_df = year_by_year_stats(df, cfg)

    stats_df.to_csv(args.outfile, index=False)
    thresholds_df.to_csv(args.thresholds_out, index=False)

    # Print a compact view
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 180)
    print("\n=== Year-by-year joint-regime evaluation ===")
    print(stats_df[[
        "year",
        "N_earnings",
        "baseline_extreme_rate",
        "n_regime",
        "regime_extreme_rate",
        "lift",
        "regime_share_of_events",
        "regime_capture_of_extremes",
    ]].to_string(index=False))

    print(f"\nWrote: {args.outfile}")
    print(f"Wrote: {args.thresholds_out}")

    if cfg.quantile_scope == "global":
        # show thresholds used
        t = thresholds_df.iloc[0].to_dict()
        print("\nThresholds (global):")
        print(f"  explosiveness >= {t['exp_thr']:.4f} ({t['exp_mode']}, q={t['exp_q']})")
        print(f"  fragility     >= {t['frag_thr']:.4f} ({t['frag_mode']}, q={t['frag_q']})")


def run_regime_eval(
    df: pd.DataFrame,
    mode: str = "standard",              # "standard" or "forward"
    train_years: str = "2018-2021",
    test_years: str = "2022-2025",
    report_train: bool = True,
    return_thresholds: bool = False,     # NEW: pipeline-friendly
):
    """
    Pipeline-friendly entrypoint.

    Default: returns stats_df only (so you can do: stage5_df = run_regime_eval(back_testing_df))
    If return_thresholds=True: returns (stats_df, thresholds_df)
    """

    cfg = RegimeConfig(
        date_col="date",
        earnings_flag_col="is_earnings_day",
        extreme_label_col="is_extreme_reaction",   # FIXED (was extreme_move)
        explosiveness_col="earnings_explosiveness_score",
        fragility_col="momentum_fragility_score",
        explosiveness_high_mode="quantile",
        fragility_high_mode="quantile",
        explosiveness_q=0.90,
        fragility_q=0.90,
        explosiveness_threshold=90.0,
        fragility_threshold=80.0,
        quantile_scope="global",
    )

    df = normalize_date(df, cfg.date_col)

    if mode == "forward":
        stats_df, thresholds_df = forward_validation_stats(
            df, cfg,
            train_years_spec=train_years,
            test_years_spec=test_years,
            report_train=report_train,
        )
    elif mode == "standard":
        stats_df, thresholds_df = year_by_year_stats(df, cfg)
    else:
        raise ValueError("mode must be 'standard' or 'forward'")

    return (stats_df, thresholds_df) if return_thresholds else stats_df
