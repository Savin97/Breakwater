

#!/usr/bin/env python3
"""
year_by_year_holds_thres_eval.py

Adds "forward validation mode":

Train period (e.g., 2016–2021):
  - compute thresholds (global quantiles or fixed thresholds) using TRAIN earnings rows only

Test period (e.g., 2022–2025):
  - apply those frozen thresholds to TEST years
  - compute year-by-year stats on TEST (and TRAIN optionally)

This is the most important credibility check:
  thresholds learned from past only, evaluated on future only.

Usage examples:

1) Forward validation with frozen quantile thresholds learned on 2016–2021, tested 2022–2025:
python year_by_year_regime_eval.py --infile data/scored.parquet \
  --mode forward \
  --train_years 2016-2021 \
  --test_years 2022-2025 \
  --exp_mode quantile --exp_q 0.90 \
  --frag_mode quantile --frag_q 0.90

2) Same, but also print TRAIN stats:
python year_by_year_regime_eval.py --infile data/scored.parquet \
  --mode forward --report_train 1 \
  --train_years 2016-2021 --test_years 2022-2025

Notes:
- Requires columns: date, is_earnings_day, is_extreme_reaction (or your label),
  earnings_explosiveness_score, momentum_fragility_score
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


# -----------------------------
# Stats helpers
# -----------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = (z * math.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def safe_rate(k: int, n: int) -> float:
    return float(k / n) if n > 0 else np.nan


def parse_years(spec: str) -> List[int]:
    """
    Parses:
      "2016-2021" -> [2016..2021]
      "2018,2019,2020" -> [2018,2019,2020]
      "2016-2018,2020,2022-2023" -> ...
    """
    years: List[int] = []
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a_i, b_i = int(a), int(b)
            if b_i < a_i:
                raise ValueError(f"Bad range: {p}")
            years.extend(list(range(a_i, b_i + 1)))
        else:
            years.append(int(p))
    years = sorted(set(years))
    return years


# -----------------------------
# Config
# -----------------------------
@dataclass
class RegimeConfig:
    date_col: str = "date"
    earnings_flag_col: str = "is_earnings_day"
    extreme_label_col: str = "is_extreme_reaction"
    explosiveness_col: str = "earnings_explosiveness_score"
    fragility_col: str = "momentum_fragility_score"

    explosiveness_high_mode: str = "quantile"  # {"quantile","threshold"}
    fragility_high_mode: str = "quantile"      # {"quantile","threshold"}

    explosiveness_q: float = 0.90
    fragility_q: float = 0.90

    explosiveness_threshold: float = 90.0
    fragility_threshold: float = 80.0


# -----------------------------
# IO / prep
# -----------------------------
def load_df(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path.lower())[1]
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if ext in [".csv"]:
        return pd.read_csv(path)
    if ext in [".feather"]:
        return pd.read_feather(path)
    raise ValueError(f"Unsupported file extension: {ext}. Use .parquet/.csv/.feather")


def normalize_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce", utc=False)
    return out


def build_earnings_df(df: pd.DataFrame, cfg: RegimeConfig) -> pd.DataFrame:
    edf = df[df[cfg.earnings_flag_col] == True].copy()
    edf = edf.dropna(subset=[cfg.date_col, cfg.extreme_label_col, cfg.explosiveness_col, cfg.fragility_col])
    edf["year"] = edf[cfg.date_col].dt.year
    return edf


# -----------------------------
# Thresholding / regime flags
# -----------------------------
def compute_thresholds(earnings_df: pd.DataFrame, cfg: RegimeConfig) -> Tuple[float, float]:
    if cfg.explosiveness_high_mode == "quantile":
        exp_thr = float(earnings_df[cfg.explosiveness_col].quantile(cfg.explosiveness_q))
    else:
        exp_thr = float(cfg.explosiveness_threshold)

    if cfg.fragility_high_mode == "quantile":
        frag_thr = float(earnings_df[cfg.fragility_col].quantile(cfg.fragility_q))
    else:
        frag_thr = float(cfg.fragility_threshold)

    return exp_thr, frag_thr


def add_regime_flags(earnings_df: pd.DataFrame, cfg: RegimeConfig, exp_thr: float, frag_thr: float) -> pd.DataFrame:
    out = earnings_df.copy()
    out["is_high_explosiveness"] = out[cfg.explosiveness_col] >= exp_thr
    out["is_high_fragility"] = out[cfg.fragility_col] >= frag_thr
    out["is_joint_regime"] = out["is_high_explosiveness"] & out["is_high_fragility"]
    return out


# -----------------------------
# Stats
# -----------------------------
def compute_yearly_table(earnings_df_with_flags: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for y, g in earnings_df_with_flags.groupby("year"):
        N = int(len(g))
        K = int(g["__label__"].sum())
        base = safe_rate(K, N)

        reg = g[g["is_joint_regime"]]
        n = int(len(reg))
        k = int(reg["__label__"].sum())
        reg_rate = safe_rate(k, n)
        lift = (reg_rate / base) if (np.isfinite(reg_rate) and np.isfinite(base) and base > 0) else np.nan

        base_ci = wilson_ci(K, N)
        reg_ci = wilson_ci(k, n)

        rows.append({
            "split": label,
            "year": int(y),
            "N_earnings": N,
            "K_extreme": K,
            "baseline_extreme_rate": base,
            "baseline_ci_low": base_ci[0],
            "baseline_ci_high": base_ci[1],
            "n_regime": n,
            "k_regime_extreme": k,
            "regime_extreme_rate": reg_rate,
            "regime_ci_low": reg_ci[0],
            "regime_ci_high": reg_ci[1],
            "lift": lift,
            "regime_share_of_events": (n / N) if N > 0 else np.nan,
            "regime_capture_of_extremes": (k / K) if K > 0 else np.nan,
        })

    out = pd.DataFrame(rows).sort_values(["split", "year"]).reset_index(drop=True)

    # Summary row per split
    g = earnings_df_with_flags
    N_all = int(len(g))
    K_all = int(g["__label__"].sum())
    base_all = safe_rate(K_all, N_all)

    reg_all = g[g["is_joint_regime"]]
    n_all = int(len(reg_all))
    k_all = int(reg_all["__label__"].sum())
    reg_rate_all = safe_rate(k_all, n_all)
    lift_all = (reg_rate_all / base_all) if (np.isfinite(reg_rate_all) and np.isfinite(base_all) and base_all > 0) else np.nan

    base_ci_all = wilson_ci(K_all, N_all)
    reg_ci_all = wilson_ci(k_all, n_all)

    summary = pd.DataFrame([{
        "split": label,
        "year": -1,
        "N_earnings": N_all,
        "K_extreme": K_all,
        "baseline_extreme_rate": base_all,
        "baseline_ci_low": base_ci_all[0],
        "baseline_ci_high": base_ci_all[1],
        "n_regime": n_all,
        "k_regime_extreme": k_all,
        "regime_extreme_rate": reg_rate_all,
        "regime_ci_low": reg_ci_all[0],
        "regime_ci_high": reg_ci_all[1],
        "lift": lift_all,
        "regime_share_of_events": (n_all / N_all) if N_all > 0 else np.nan,
        "regime_capture_of_extremes": (k_all / K_all) if K_all > 0 else np.nan,
    }])

    return pd.concat([out, summary], ignore_index=True)


def run_forward_validation(
    earnings_df: pd.DataFrame,
    cfg: RegimeConfig,
    train_years: List[int],
    test_years: List[int],
    report_train: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Attach label column with a standard name (binary 0/1)
    edf = earnings_df.copy()
    edf["__label__"] = edf[cfg.extreme_label_col].astype(int)

    train = edf[edf["year"].isin(train_years)].copy()
    test = edf[edf["year"].isin(test_years)].copy()

    if len(train) == 0:
        raise ValueError("No rows in train split. Check --train_years.")
    if len(test) == 0:
        raise ValueError("No rows in test split. Check --test_years.")

    exp_thr, frag_thr = compute_thresholds(train, cfg)

    out_tables = []

    if report_train:
        train_f = add_regime_flags(train, cfg, exp_thr, frag_thr)
        out_tables.append(compute_yearly_table(train_f, label="TRAIN"))

    test_f = add_regime_flags(test, cfg, exp_thr, frag_thr)
    out_tables.append(compute_yearly_table(test_f, label="TEST"))

    stats_df = pd.concat(out_tables, ignore_index=True)

    thresholds_df = pd.DataFrame([{
        "train_years": ",".join(map(str, train_years)),
        "test_years": ",".join(map(str, test_years)),
        "exp_thr": exp_thr,
        "frag_thr": frag_thr,
        "exp_mode": cfg.explosiveness_high_mode,
        "frag_mode": cfg.fragility_high_mode,
        "exp_q": cfg.explosiveness_q,
        "frag_q": cfg.fragility_q,
        "exp_fixed_thr": cfg.explosiveness_threshold,
        "frag_fixed_thr": cfg.fragility_threshold,
    }])

    return stats_df, thresholds_df


def run_standard_year_by_year(earnings_df: pd.DataFrame, cfg: RegimeConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    edf = earnings_df.copy()
    edf["__label__"] = edf[cfg.extreme_label_col].astype(int)

    exp_thr, frag_thr = compute_thresholds(edf, cfg)
    edf = add_regime_flags(edf, cfg, exp_thr, frag_thr)

    stats_df = compute_yearly_table(edf, label="ALL")

    thresholds_df = pd.DataFrame([{
        "exp_thr": exp_thr,
        "frag_thr": frag_thr,
        "exp_mode": cfg.explosiveness_high_mode,
        "frag_mode": cfg.fragility_high_mode,
        "exp_q": cfg.explosiveness_q,
        "frag_q": cfg.fragility_q,
    }])

    return stats_df, thresholds_df




# def main(df):
#     p = argparse.ArgumentParser()
#     p.add_argument("--outfile", default="year_by_year_regime_eval.csv", help="CSV output path.")
#     p.add_argument("--thresholds_out", default="year_by_year_thresholds.csv", help="Thresholds output path.")

#     p.add_argument("--mode", choices=["standard", "forward"], default="standard",
#                    help="standard: thresholds from all years; forward: thresholds from TRAIN only, evaluated on TEST.")

#     p.add_argument("--train_years", default="2016-2021", help="Years for train split (forward mode).")
#     p.add_argument("--test_years", default="2022-2025", help="Years for test split (forward mode).")
#     p.add_argument("--report_train", type=int, default=0, help="In forward mode, also output TRAIN year-by-year stats (1/0).")

#     # Columns
#     p.add_argument("--date_col", default="date")
#     p.add_argument("--earnings_flag_col", default="is_earnings_day")
#     p.add_argument("--extreme_label_col", default="is_extreme_reaction")
#     p.add_argument("--explosiveness_col", default="earnings_explosiveness_score")
#     p.add_argument("--fragility_col", default="momentum_fragility_score")

#     # Regime definition
#     p.add_argument("--exp_mode", choices=["quantile", "threshold"], default="quantile")
#     p.add_argument("--frag_mode", choices=["quantile", "threshold"], default="quantile")
#     p.add_argument("--exp_q", type=float, default=0.90)
#     p.add_argument("--frag_q", type=float, default=0.90)
#     p.add_argument("--exp_thr", type=float, default=90.0)
#     p.add_argument("--frag_thr", type=float, default=80.0)

#     args = p.parse_args()

#     cfg = RegimeConfig(
#         date_col=args.date_col,
#         earnings_flag_col=args.earnings_flag_col,
#         extreme_label_col=args.extreme_label_col,
#         explosiveness_col=args.explosiveness_col,
#         fragility_col=args.fragility_col,
#         explosiveness_high_mode=args.exp_mode,
#         fragility_high_mode=args.frag_mode,
#         explosiveness_q=args.exp_q,
#         fragility_q=args.frag_q,
#         explosiveness_threshold=args.exp_thr,
#         fragility_threshold=args.frag_thr,
#     )

#     df = normalize_date(df, cfg.date_col)

#     required = [cfg.date_col, cfg.earnings_flag_col, cfg.extreme_label_col, cfg.explosiveness_col, cfg.fragility_col]
#     missing = [c for c in required if c not in df.columns]
#     if missing:
#         raise ValueError(f"Missing required columns: {missing}")

#     earnings_df = build_earnings_df(df, cfg)

#     if args.mode == "forward":
#         train_years = parse_years(args.train_years)
#         test_years = parse_years(args.test_years)
#         report_train = bool(args.report_train)

#         stats_df, thresholds_df = run_forward_validation(
#             earnings_df=earnings_df,
#             cfg=cfg,
#             train_years=train_years,
#             test_years=test_years,
#             report_train=report_train,
#         )
#     else:
#         stats_df, thresholds_df = run_standard_year_by_year(earnings_df, cfg)

#     stats_df.to_csv(args.outfile, index=False)
#     thresholds_df.to_csv(args.thresholds_out, index=False)

#     # Print a compact view
#     pd.set_option("display.max_rows", 300)
#     pd.set_option("display.width", 220)

#     print("\n=== Year-by-year joint-regime evaluation ===")
#     cols = [
#         "split", "year",
#         "N_earnings", "baseline_extreme_rate",
#         "n_regime", "regime_extreme_rate",
#         "lift",
#         "regime_share_of_events", "regime_capture_of_extremes",
#     ]
#     print(stats_df[cols].to_string(index=False))

#     print(f"\nWrote: {args.outfile}")
#     print(f"Wrote: {args.thresholds_out}")

#     # Show frozen thresholds in forward mode
#     if args.mode == "forward":
#         t = thresholds_df.iloc[0].to_dict()
#         print("\nFrozen thresholds (learned on TRAIN):")
#         print(f"  TRAIN years: {t['train_years']}")
#         print(f"  TEST years : {t['test_years']}")
#         print(f"  explosiveness >= {t['exp_thr']:.4f} (mode={t['exp_mode']}, q={t['exp_q']})")
#         print(f"  fragility     >= {t['frag_thr']:.4f} (mode={t['frag_mode']}, q={t['frag_q']})")
#     else:
#         t = thresholds_df.iloc[0].to_dict()
#         print("\nThresholds (computed on ALL):")
#         print(f"  explosiveness >= {t['exp_thr']:.4f} (mode={t['exp_mode']}, q={t['exp_q']})")
#         print(f"  fragility     >= {t['frag_thr']:.4f} (mode={t['frag_mode']}, q={t['frag_q']})")


