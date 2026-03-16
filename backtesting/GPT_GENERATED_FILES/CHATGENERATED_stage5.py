

# pipeline/stage5_backtest_metrics.py
"""
Stage 5 - Backtest Metrics (Part A: Signal validity)

This stage focuses on "Does the score correspond to realized earnings risk?"
It computes:
  - Bucket (quantile) curves for event rates + magnitude stats
  - Lift + capture in top X%
  - Rank-quality metrics (Spearman, AUC)
  - Optional probability calibration metrics (Brier, reliability buckets) if you provide probs

No external deps beyond pandas/numpy.

Expected inputs:
  - DataFrame with earnings rows flagged:
      is_earnings_day (bool/int)
  - Realized move column(s):
      reaction_3d OR abs_reaction_3d
  - Binary event labels OR enough info to build them:
      is_large_reaction, is_extreme_reaction (preferred)
      or earnings_move_bucket (0/1/2) (fallback)
  - Score columns (examples):
      timing_danger, earnings_explosiveness_score, vol_expansion_score, momentum_fragility_score, proximity_score

Outputs:
  - dict of metrics
  - bucket tables per score
  - optional CSV/JSON artifacts under backtesting/artifacts/
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import json



# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class BacktestAConfig:
    # Which columns to evaluate as "scores"
    score_cols: Tuple[str, ...] = (
        "timing_danger",
        "earnings_explosiveness_score",
        "vol_expansion_score",
        "momentum_fragility_score",
        "proximity_score",
    )

    # Earnings event filters
    earnings_flag_col: str = "is_earnings_day"

    # Realized reaction columns (use whichever exists)
    reaction_col_preferred: str = "reaction_3d"
    abs_reaction_col: str = "abs_reaction_3d"

    # Label columns (preferred); fallback supported
    large_label_col: str = "is_large_reaction"
    extreme_label_col: str = "is_extreme_reaction"
    bucket_label_col: str = "earnings_move_bucket"  # fallback if labels not present

    # Bucketing controls
    n_buckets: int = 10  # deciles
    min_bucket_n: int = 25  # guardrails for noisy buckets

    # Top-X concentration metrics
    top_fracs: Tuple[float, ...] = (0.05, 0.10, 0.20)

    # Artifact output
    artifacts_dir: str = "backtesting/artifacts"
    write_artifacts: bool = True


# -----------------------------
# Utilities
# -----------------------------

def _ensure_bool01(s: pd.Series) -> pd.Series:
    """Convert to 0/1 int safely."""
    if s.dtype == bool:
        return s.astype(int)
    # if floats with NaN, treat NaN as 0 for labels (but you may prefer to drop)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def _get_abs_reaction(df: pd.DataFrame, cfg: BacktestAConfig) -> pd.Series:
    if cfg.abs_reaction_col in df.columns:
        return pd.to_numeric(df[cfg.abs_reaction_col], errors="coerce")
    if cfg.reaction_col_preferred in df.columns:
        return pd.to_numeric(df[cfg.reaction_col_preferred], errors="coerce").abs()
    raise KeyError(
        f"Need either '{cfg.abs_reaction_col}' or '{cfg.reaction_col_preferred}' in df."
    )


def _get_labels(df: pd.DataFrame, cfg: BacktestAConfig) -> Tuple[pd.Series, pd.Series]:
    """
    Return (large_label, extreme_label) as 0/1 int series.
    Preferred: is_large_reaction / is_extreme_reaction
    Fallback: earnings_move_bucket (0=small,1=large_plus,2=extreme)
    """
    if cfg.large_label_col in df.columns and cfg.extreme_label_col in df.columns:
        large = _ensure_bool01(df[cfg.large_label_col])
        extreme = _ensure_bool01(df[cfg.extreme_label_col])
        return large, extreme

    if cfg.bucket_label_col in df.columns:
        b = pd.to_numeric(df[cfg.bucket_label_col], errors="coerce")
        # Convention in your code: 0/1/2 (1=large_plus,2=extreme)
        large = (b >= 1).fillna(False).astype(int)
        extreme = (b >= 2).fillna(False).astype(int)
        return large, extreme

    raise KeyError(
        f"Need label cols ('{cfg.large_label_col}','{cfg.extreme_label_col}') "
        f"or fallback '{cfg.bucket_label_col}'."
    )


def _filter_earnings_rows(df: pd.DataFrame, cfg: BacktestAConfig) -> pd.DataFrame:
    if cfg.earnings_flag_col not in df.columns:
        raise KeyError(f"Missing earnings flag column '{cfg.earnings_flag_col}'")
    mask = df[cfg.earnings_flag_col].astype(bool)
    out = df.loc[mask].copy()
    return out


def _safe_quantile_buckets(scores: pd.Series, n_buckets: int) -> pd.Series:
    """
    Quantile bucketing robust to duplicate values. Uses rank-based qcut.
    Returns buckets 1..n_buckets (int), NaN where score is NaN.
    """
    s = pd.to_numeric(scores, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(index=s.index, dtype="float64")
    # rank to avoid qcut failing on many duplicates
    r = s.rank(method="average", na_option="keep")
    # qcut on rank; duplicates='drop' can reduce #bins if needed
    try:
        b = pd.qcut(r, q=n_buckets, labels=False, duplicates="drop")
        return (b + 1).astype("float")
    except ValueError:
        # if still fails, return a single bucket
        return pd.Series(np.where(s.notna(), 1.0, np.nan), index=s.index)


def _spearman(x: pd.Series, y: pd.Series) -> float:
    """Spearman rho using pandas rank corr. Returns nan if insufficient."""
    xs = pd.to_numeric(x, errors="coerce")
    ys = pd.to_numeric(y, errors="coerce")
    ok = xs.notna() & ys.notna()
    if ok.sum() < 3:
        return float("nan")
    return float(xs[ok].rank().corr(ys[ok].rank(), method="pearson"))


def _auc_from_scores(scores: pd.Series, labels01: pd.Series) -> float:
    """
    AUC computed via rank statistic (equivalent to Mann–Whitney U).
    No sklearn required.

    Returns nan if not enough positives/negatives.
    """
    s = pd.to_numeric(scores, errors="coerce")
    y = _ensure_bool01(labels01)
    ok = s.notna() & y.notna()
    s = s[ok]
    y = y[ok].astype(int)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = s.rank(method="average")
    sum_ranks_pos = float(ranks[y == 1].sum())
    # U statistic for positives
    u_pos = sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)
    auc = u_pos / (n_pos * n_neg)
    return float(auc)


def _brier_score(probs: pd.Series, labels01: pd.Series) -> float:
    p = pd.to_numeric(probs, errors="coerce")
    y = pd.to_numeric(labels01, errors="coerce")
    ok = p.notna() & y.notna()
    if ok.sum() == 0:
        return float("nan")
    p = p[ok].clip(0, 1)
    y = y[ok].astype(float)
    return float(np.mean((p - y) ** 2))


# -----------------------------
# Part A: Signal validity metrics
# -----------------------------

def bucket_curve_table(
    earnings_df: pd.DataFrame,
    score_col: str,
    abs_move: pd.Series,
    large_label: pd.Series,
    extreme_label: pd.Series,
    cfg: BacktestAConfig,
) -> pd.DataFrame:
    """
    Decile/quantile table:
      - n
      - event rates (large/extreme)
      - abs move stats (mean/median/p75/p90/p95)
    """
    s = pd.to_numeric(earnings_df[score_col], errors="coerce")
    buckets = _safe_quantile_buckets(s, cfg.n_buckets)

    tmp = pd.DataFrame(
        {
            "bucket": buckets,
            "score": s,
            "abs_move": abs_move,
            "is_large": large_label,
            "is_extreme": extreme_label,
        }
    ).dropna(subset=["bucket"])

    # bucket stats
    def q(p: float):
        return lambda x: float(np.nanquantile(x, p))

    agg = tmp.groupby("bucket").agg(
        n=("abs_move", "size"),
        score_min=("score", "min"),
        score_max=("score", "max"),
        p_large=("is_large", "mean"),
        p_extreme=("is_extreme", "mean"),
        abs_mean=("abs_move", "mean"),
        abs_median=("abs_move", "median"),
        abs_p75=("abs_move", q(0.75)),
        abs_p90=("abs_move", q(0.90)),
        abs_p95=("abs_move", q(0.95)),
    ).reset_index()

    # Add monotonic diagnostics (simple)
    # (These are not hard pass/fail here; you can enforce later in Part C.)
    agg["p_large_diff_prev"] = agg["p_large"].diff()
    agg["p_extreme_diff_prev"] = agg["p_extreme"].diff()
    agg["abs_mean_diff_prev"] = agg["abs_mean"].diff()

    # Flag small-N buckets
    agg["small_n_flag"] = agg["n"] < cfg.min_bucket_n

    # Ensure bucket order as int
    agg["bucket"] = agg["bucket"].astype(int)
    agg = agg.sort_values("bucket").reset_index(drop=True)
    return agg


def top_x_concentration(
    earnings_df: pd.DataFrame,
    score_col: str,
    large_label: pd.Series,
    extreme_label: pd.Series,
    cfg: BacktestAConfig,
) -> pd.DataFrame:
    """
    For top X% highest scores:
      - n_top
      - event_rate_top
      - lift_top (rate_top / rate_all)
      - capture_top (events_in_top / total_events)
    """
    s = pd.to_numeric(earnings_df[score_col], errors="coerce")
    ok = s.notna()
    s = s[ok]
    large = large_label.loc[s.index]
    extreme = extreme_label.loc[s.index]

    out_rows = []
    base_large = float(large.mean()) if len(large) else float("nan")
    base_ext = float(extreme.mean()) if len(extreme) else float("nan")
    total_large = float(large.sum())
    total_ext = float(extreme.sum())

    for frac in cfg.top_fracs:
        if len(s) == 0:
            out_rows.append(
                {
                    "top_frac": frac,
                    "n_top": 0,
                    "large_rate_top": np.nan,
                    "large_lift": np.nan,
                    "large_capture": np.nan,
                    "extreme_rate_top": np.nan,
                    "extreme_lift": np.nan,
                    "extreme_capture": np.nan,
                }
            )
            continue

        k = int(np.ceil(len(s) * frac))
        top_idx = s.sort_values(ascending=False).head(k).index

        large_top = large.loc[top_idx]
        ext_top = extreme.loc[top_idx]

        large_rate_top = float(large_top.mean()) if len(large_top) else float("nan")
        ext_rate_top = float(ext_top.mean()) if len(ext_top) else float("nan")

        out_rows.append(
            {
                "top_frac": frac,
                "n_top": int(k),
                "large_rate_top": large_rate_top,
                "large_lift": (large_rate_top / base_large) if base_large and base_large > 0 else np.nan,
                "large_capture": (float(large_top.sum()) / total_large) if total_large and total_large > 0 else np.nan,
                "extreme_rate_top": ext_rate_top,
                "extreme_lift": (ext_rate_top / base_ext) if base_ext and base_ext > 0 else np.nan,
                "extreme_capture": (float(ext_top.sum()) / total_ext) if total_ext and total_ext > 0 else np.nan,
            }
        )

    return pd.DataFrame(out_rows)


def compute_signal_validity_metrics(
    df: pd.DataFrame,
    cfg: Optional[BacktestAConfig] = None,
    prob_cols: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, object]:
    """
    Main Part-A entrypoint.

    prob_cols (optional):
      {
        "timing_danger": {"large": "p_large_pred", "extreme": "p_extreme_pred"},
        ...
      }
    If provided, compute Brier score + reliability tables (bucketed by predicted prob).

    Returns:
      {
        "meta": {...},
        "per_score": {
            score_col: {
                "rank": {...},
                "top_x": DataFrame,
                "bucket_curve": DataFrame,
                "calibration": {... optional ...}
            }
        }
      }
    """
    cfg = cfg or BacktestAConfig()

    earnings_df = _filter_earnings_rows(df, cfg)
    abs_move = _get_abs_reaction(earnings_df, cfg)
    large_label, extreme_label = _get_labels(earnings_df, cfg)

    # basic sanity counts
    meta = {
        "n_earnings_rows": int(len(earnings_df)),
        "base_large_rate": float(large_label.mean()) if len(large_label) else float("nan"),
        "base_extreme_rate": float(extreme_label.mean()) if len(extreme_label) else float("nan"),
        "score_cols_evaluated": [],
        "missing_score_cols": [],
    }

    results: Dict[str, object] = {"meta": meta, "per_score": {}}

    for score_col in cfg.score_cols:
        if score_col not in earnings_df.columns:
            meta["missing_score_cols"].append(score_col)
            continue

        meta["score_cols_evaluated"].append(score_col)
        s = pd.to_numeric(earnings_df[score_col], errors="coerce")

        # Rank-quality metrics
        rank_metrics = {
            "spearman_score_vs_abs_move": _spearman(s, abs_move),
            "auc_large_move": _auc_from_scores(s, large_label),
            "auc_extreme_move": _auc_from_scores(s, extreme_label),
        }

        # Bucket curve
        bucket_tbl = bucket_curve_table(
            earnings_df=earnings_df,
            score_col=score_col,
            abs_move=abs_move,
            large_label=large_label,
            extreme_label=extreme_label,
            cfg=cfg,
        )

        # Top-X concentration
        top_tbl = top_x_concentration(
            earnings_df=earnings_df,
            score_col=score_col,
            large_label=large_label,
            extreme_label=extreme_label,
            cfg=cfg,
        )

        per_score_obj: Dict[str, object] = {
            "rank": rank_metrics,
            "bucket_curve": bucket_tbl,
            "top_x": top_tbl,
        }

        # Optional probability calibration (if you have predicted probs)
        if prob_cols and score_col in prob_cols:
            cal = {}
            for evt, pcol in prob_cols[score_col].items():
                if pcol in earnings_df.columns:
                    y = large_label if evt == "large" else extreme_label
                    p = earnings_df[pcol]
                    cal[f"brier_{evt}"] = _brier_score(p, y)

                    # reliability table: bin by predicted prob deciles
                    rel_bins = _safe_quantile_buckets(pd.to_numeric(p, errors="coerce"), cfg.n_buckets)
                    rel = pd.DataFrame({"bin": rel_bins, "p": p, "y": y}).dropna(subset=["bin"])
                    rel_tbl = (
                        rel.groupby("bin")
                        .agg(n=("y", "size"), p_pred=("p", "mean"), p_real=("y", "mean"))
                        .reset_index()
                        .sort_values("bin")
                    )
                    cal[f"reliability_{evt}"] = rel_tbl
            if cal:
                per_score_obj["calibration"] = cal

        results["per_score"][score_col] = per_score_obj

    return results


def write_part_a_artifacts(
    metrics: Dict[str, object],
    cfg: Optional[BacktestAConfig] = None,
) -> None:
    """
    Write Part-A artifacts to CSV/JSON.
    - summary json
    - per-score bucket curves CSV
    - per-score top-x CSV
    - optional reliability CSVs
    """
    cfg = cfg or BacktestAConfig()
    if not cfg.write_artifacts:
        return

    out_dir = Path(cfg.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON summary without embedding huge tables
    summary = {
        "meta": metrics.get("meta", {}),
        "per_score_rank": {
            k: v.get("rank", {}) for k, v in metrics.get("per_score", {}).items()
        },
    }
    (out_dir / "part_a_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Tables
    for score_col, obj in metrics.get("per_score", {}).items():
        bucket_tbl = obj.get("bucket_curve")
        top_tbl = obj.get("top_x")

        if isinstance(bucket_tbl, pd.DataFrame):
            bucket_tbl.to_csv(out_dir / f"{score_col}__bucket_curve.csv", index=False)
        if isinstance(top_tbl, pd.DataFrame):
            top_tbl.to_csv(out_dir / f"{score_col}__top_x.csv", index=False)

        cal = obj.get("calibration", {})
        if isinstance(cal, dict):
            for k, v in cal.items():
                if k.startswith("reliability_") and isinstance(v, pd.DataFrame):
                    v.to_csv(out_dir / f"{score_col}__{k}.csv", index=False)


# -----------------------------
# Stage wrapper (only Part A for now)
# -----------------------------

def stage5_part_a(stage4_df: pd.DataFrame, cfg: Optional[BacktestAConfig] = None) -> pd.DataFrame:
    """
    Pipeline hook: call after stage4 (so labels + abs_reaction exist).
    Computes Part-A metrics + writes artifacts.

    Returns df unchanged (metrics are side artifacts).
    """
    cfg = cfg or BacktestAConfig()
    metrics = compute_signal_validity_metrics(stage4_df, cfg=cfg)
    write_part_a_artifacts(metrics, cfg=cfg)

    # Print compact console summary (PM-friendly)
    meta = metrics.get("meta", {})
    print("\n[Stage5 Part A] Signal validity summary")
    print("Earnings rows:", meta.get("n_earnings_rows"))
    print("Base large rate:", meta.get("base_large_rate"))
    print("Base extreme rate:", meta.get("base_extreme_rate"))

    for score_col, obj in metrics.get("per_score", {}).items():
        r = obj.get("rank", {})
        print(
            f"- {score_col}: spearman(abs_move)={r.get('spearman_score_vs_abs_move'):.3f} "
            f"AUC_large={r.get('auc_large_move'):.3f} AUC_extreme={r.get('auc_extreme_move'):.3f}"
        )

    return stage4_df





# pipeline/stage5_backtest_metrics.py
"""
Stage 5 - Backtest Metrics (Part B: Decision usefulness + conditional regime tests)

Part B answers: "If I use the score to position, does it reduce realized earnings risk?"

It computes:
  1) Policy backtests:
       - "Avoid / Reduce" if score >= threshold (quantile or absolute)
       - Compare risk outcomes vs baseline / vs hold set
       - Key: tail reduction + false negatives (missed extreme events)

  2) Joint regime tests (2-way and optional 3-way):
       - e.g. high timing_danger AND high explosiveness
       - event rates, lifts, sample sizes

  3) Proper universe selection:
       - earnings_day universe for event labels based on post-earnings reactions
       - pre-earnings universe for proximity-based positioning:
           use is_earnings_week or is_earnings_window

Dependencies: pandas/numpy only.

This file is intended to live alongside Part A in the same module.
If you kept Part A in a separate module, copy the shared utilities.
"""



# ============================================================
# Shared (minimal) utilities — must match Part A assumptions
# ============================================================

def _ensure_bool01(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def _get_abs_reaction(df: pd.DataFrame, abs_col: str = "abs_reaction_3d", reaction_col: str = "reaction_3d") -> pd.Series:
    if abs_col in df.columns:
        return pd.to_numeric(df[abs_col], errors="coerce")
    if reaction_col in df.columns:
        return pd.to_numeric(df[reaction_col], errors="coerce").abs()
    raise KeyError(f"Need either '{abs_col}' or '{reaction_col}' in df.")


def _get_labels(
    df: pd.DataFrame,
    large_label_col: str = "is_large_reaction",
    extreme_label_col: str = "is_extreme_reaction",
    bucket_label_col: str = "earnings_move_bucket",
) -> Tuple[pd.Series, pd.Series]:
    if large_label_col in df.columns and extreme_label_col in df.columns:
        return _ensure_bool01(df[large_label_col]), _ensure_bool01(df[extreme_label_col])
    if bucket_label_col in df.columns:
        b = pd.to_numeric(df[bucket_label_col], errors="coerce")
        large = (b >= 1).fillna(False).astype(int)
        extreme = (b >= 2).fillna(False).astype(int)
        return large, extreme
    raise KeyError(
        f"Need label cols ('{large_label_col}','{extreme_label_col}') or fallback '{bucket_label_col}'."
    )


def _filter_rows(df: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    if flag_col not in df.columns:
        raise KeyError(f"Missing flag column '{flag_col}'")
    return df.loc[df[flag_col].astype(bool)].copy()


def _quantile_threshold(scores: pd.Series, q: float) -> float:
    s = pd.to_numeric(scores, errors="coerce").dropna()
    if len(s) == 0:
        return float("nan")
    return float(np.nanquantile(s.to_numpy(), q))


def _tail_stats(x: pd.Series) -> Dict[str, float]:
    v = pd.to_numeric(x, errors="coerce").dropna()
    if len(v) == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "p75": np.nan, "p90": np.nan, "p95": np.nan}
    arr = v.to_numpy()
    return {
        "n": float(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p75": float(np.nanquantile(arr, 0.75)),
        "p90": float(np.nanquantile(arr, 0.90)),
        "p95": float(np.nanquantile(arr, 0.95)),
    }


# ============================================================
# Part B config
# ============================================================

@dataclass(frozen=True)
class BacktestBConfig:
    # Universe flags
    earnings_day_flag_col: str = "is_earnings_day"
    pre_earnings_flag_col: str = "is_earnings_window"  # better for proximity / positioning
    # You can switch to "is_earnings_week" if you prefer.

    # Reactions / labels
    abs_reaction_col: str = "abs_reaction_3d"
    reaction_col: str = "reaction_3d"
    large_label_col: str = "is_large_reaction"
    extreme_label_col: str = "is_extreme_reaction"
    bucket_label_col: str = "earnings_move_bucket"

    # Policies: evaluate thresholds at these quantiles (top X%)
    # Example: q=0.90 means "flag top 10% highest risk"
    policy_quantiles: Tuple[float, ...] = (0.80, 0.90, 0.95)

    # Scores for policy tests
    policy_score_cols_day: Tuple[str, ...] = (
        "timing_danger",
        "earnings_explosiveness_score",
        "momentum_fragility_score",
        "vol_expansion_score",
    )

    # Scores for pre-earnings policy tests (positioning BEFORE earnings)
    policy_score_cols_pre: Tuple[str, ...] = (
        "timing_danger",
        "proximity_score",
        "momentum_fragility_score",
        "vol_expansion_score",
    )

    # Joint regime tests (2-way high/high)
    # Each pair is (score_a, score_b)
    joint_pairs: Tuple[Tuple[str, str], ...] = (
        ("timing_danger", "earnings_explosiveness_score"),
        ("timing_danger", "momentum_fragility_score"),
        ("earnings_explosiveness_score", "momentum_fragility_score"),
        ("timing_danger", "vol_expansion_score"),
    )

    # "High" definition for regime tests (top quantile)
    regime_high_q: float = 0.90

    # Guardrails
    min_group_n: int = 30  # avoid noisy regime cells

    # Output
    artifacts_dir: str = "backtesting/artifacts"
    write_artifacts: bool = True


# ============================================================
# Part B core computations
# ============================================================

def policy_backtest_one_score(
    df_events: pd.DataFrame,
    score_col: str,
    abs_move: pd.Series,
    large_label: pd.Series,
    extreme_label: pd.Series,
    quantiles: Tuple[float, ...],
) -> pd.DataFrame:
    """
    For each threshold quantile q:
      - define FLAG if score >= q-quantile threshold
      - report:
          flagged rate
          event rates in flagged vs hold
          tail stats of abs_move in flagged vs hold
          capture (recall) of events by flagged
          miss rate (false negatives) for events under policy (events not flagged)
    """
    s = pd.to_numeric(df_events[score_col], errors="coerce")
    ok = s.notna()
    if ok.sum() == 0:
        return pd.DataFrame()

    s = s[ok]
    abs_move = abs_move.loc[s.index]
    large_label = large_label.loc[s.index]
    extreme_label = extreme_label.loc[s.index]

    base_large = float(large_label.mean())
    base_ext = float(extreme_label.mean())
    print(len(s))
    out = []
    for q in quantiles:
        thr = _quantile_threshold(s, q)
        print(f"threshold for q={q:.2f} is {thr:.4f}")
        flag = (s >= thr).astype(int)
        idx_flag = flag[flag == 1].index
        idx_hold = flag[flag == 0].index

        # Rates
        large_flag = float(large_label.loc[idx_flag].mean()) if len(idx_flag) else np.nan
        large_hold = float(large_label.loc[idx_hold].mean()) if len(idx_hold) else np.nan
        ext_flag = float(extreme_label.loc[idx_flag].mean()) if len(idx_flag) else np.nan
        ext_hold = float(extreme_label.loc[idx_hold].mean()) if len(idx_hold) else np.nan

        # Captures
        total_large = float(large_label.sum())
        total_ext = float(extreme_label.sum())
        cap_large = float(large_label.loc[idx_flag].sum() / total_large) if total_large > 0 else np.nan
        cap_ext = float(extreme_label.loc[idx_flag].sum() / total_ext) if total_ext > 0 else np.nan

        # Miss rates (false negatives among events)
        # FN rate among extreme events: % of extreme events not flagged
        if total_ext > 0:
            fn_ext = float(extreme_label.loc[idx_hold].sum() / total_ext)
        else:
            fn_ext = np.nan
        if total_large > 0:
            fn_large = float(large_label.loc[idx_hold].sum() / total_large)
        else:
            fn_large = np.nan

        # Tail stats
        stats_flag = _tail_stats(abs_move.loc[idx_flag])
        stats_hold = _tail_stats(abs_move.loc[idx_hold])
        stats_all = _tail_stats(abs_move)
        print("flag_rate", float(len(idx_flag) / len(s)))

        out.append(
            {
                "score_col": score_col,
                "q_threshold": q,
                "threshold_value": thr,
                "n_all": int(stats_all["n"]),
                "n_flag": int(stats_flag["n"]),
                "n_hold": int(stats_hold["n"]),
                "flag_rate": float(len(idx_flag) / len(s)),

                "base_large": base_large,
                "base_extreme": base_ext,

                "p_large_flag": large_flag,
                "p_large_hold": large_hold,
                "p_extreme_flag": ext_flag,
                "p_extreme_hold": ext_hold,

                "lift_large_flag": (large_flag / base_large) if base_large > 0 else np.nan,
                "lift_extreme_flag": (ext_flag / base_ext) if base_ext > 0 else np.nan,

                "capture_large": cap_large,
                "capture_extreme": cap_ext,
                "fn_rate_large": fn_large,
                "fn_rate_extreme": fn_ext,

                # tail reduction: compare hold tail to overall tail
                "abs_p95_all": stats_all["p95"],
                "abs_p95_hold": stats_hold["p95"],
                "abs_p95_flag": stats_flag["p95"],

                "abs_mean_all": stats_all["mean"],
                "abs_mean_hold": stats_hold["mean"],
                "abs_mean_flag": stats_flag["mean"],
            }
        )

    return pd.DataFrame(out)


def policy_backtests(
    df_events: pd.DataFrame,
    cfg: BacktestBConfig,
    score_cols: Tuple[str, ...],
    universe_name: str,
) -> Dict[str, object]:
    abs_move = _get_abs_reaction(df_events, cfg.abs_reaction_col, cfg.reaction_col)
    large_label, extreme_label = _get_labels(df_events, cfg.large_label_col, cfg.extreme_label_col, cfg.bucket_label_col)

    per_score_tables = []
    for sc in score_cols:
        if sc not in df_events.columns:
            continue
        tbl = policy_backtest_one_score(
            df_events=df_events,
            score_col=sc,
            abs_move=abs_move,
            large_label=large_label,
            extreme_label=extreme_label,
            quantiles=cfg.policy_quantiles,
        )
        if len(tbl):
            tbl.insert(0, "universe", universe_name)
            per_score_tables.append(tbl)

    out_tbl = pd.concat(per_score_tables, ignore_index=True) if per_score_tables else pd.DataFrame()
    meta = {
        "universe": universe_name,
        "n_events": int(len(df_events)),
        "base_large_rate": float(large_label.mean()) if len(large_label) else np.nan,
        "base_extreme_rate": float(extreme_label.mean()) if len(extreme_label) else np.nan,
        "score_cols": [c for c in score_cols if c in df_events.columns],
    }
    return {"meta": meta, "policy_table": out_tbl}


def joint_regime_table(
    df_events: pd.DataFrame,
    score_a: str,
    score_b: str,
    abs_move: pd.Series,
    large_label: pd.Series,
    extreme_label: pd.Series,
    high_q: float,
    min_n: int,
) -> pd.DataFrame:
    """
    2x2 regime table using "high" thresholds:
      A_high? x B_high? -> cell metrics
    """
    a = pd.to_numeric(df_events[score_a], errors="coerce")
    b = pd.to_numeric(df_events[score_b], errors="coerce")

    ok = a.notna() & b.notna()
    if ok.sum() == 0:
        return pd.DataFrame()

    a = a[ok]
    b = b[ok]
    abs_move = abs_move.loc[a.index]
    large_label = large_label.loc[a.index]
    extreme_label = extreme_label.loc[a.index]

    thr_a = _quantile_threshold(a, high_q)
    thr_b = _quantile_threshold(b, high_q)

    a_high = (a >= thr_a).astype(int)
    b_high = (b >= thr_b).astype(int)

    tmp = pd.DataFrame(
        {
            "a_high": a_high,
            "b_high": b_high,
            "abs_move": abs_move,
            "is_large": large_label,
            "is_extreme": extreme_label,
        }
    )

    base_large = float(large_label.mean())
    base_ext = float(extreme_label.mean())

    rows = []
    for ah in [0, 1]:
        for bh in [0, 1]:
            cell = tmp[(tmp["a_high"] == ah) & (tmp["b_high"] == bh)]
            n = int(len(cell))
            if n == 0:
                continue

            pL = float(cell["is_large"].mean())
            pE = float(cell["is_extreme"].mean())
            ts = _tail_stats(cell["abs_move"])

            rows.append(
                {
                    "score_a": score_a,
                    "score_b": score_b,
                    "high_q": high_q,
                    "thr_a": thr_a,
                    "thr_b": thr_b,
                    "a_high": ah,
                    "b_high": bh,
                    "n": n,
                    "small_n_flag": n < min_n,
                    "p_large": pL,
                    "p_extreme": pE,
                    "lift_large": (pL / base_large) if base_large > 0 else np.nan,
                    "lift_extreme": (pE / base_ext) if base_ext > 0 else np.nan,
                    "abs_mean": ts["mean"],
                    "abs_p95": ts["p95"],
                }
            )

    out = pd.DataFrame(rows).sort_values(["a_high", "b_high"]).reset_index(drop=True)
    return out


def joint_regime_tests(
    df_events: pd.DataFrame,
    cfg: BacktestBConfig,
    universe_name: str,
) -> Dict[str, object]:
    abs_move = _get_abs_reaction(df_events, cfg.abs_reaction_col, cfg.reaction_col)
    large_label, extreme_label = _get_labels(df_events, cfg.large_label_col, cfg.extreme_label_col, cfg.bucket_label_col)

    tables = []
    for a, b in cfg.joint_pairs:
        if a not in df_events.columns or b not in df_events.columns:
            continue
        tbl = joint_regime_table(
            df_events=df_events,
            score_a=a,
            score_b=b,
            abs_move=abs_move,
            large_label=large_label,
            extreme_label=extreme_label,
            high_q=cfg.regime_high_q,
            min_n=cfg.min_group_n,
        )
        if len(tbl):
            tbl.insert(0, "universe", universe_name)
            tables.append(tbl)

    out_tbl = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    meta = {
        "universe": universe_name,
        "n_events": int(len(df_events)),
        "base_large_rate": float(large_label.mean()) if len(large_label) else np.nan,
        "base_extreme_rate": float(extreme_label.mean()) if len(extreme_label) else np.nan,
        "pairs_tested": [(a, b) for (a, b) in cfg.joint_pairs if a in df_events.columns and b in df_events.columns],
        "high_q": cfg.regime_high_q,
    }
    return {"meta": meta, "joint_table": out_tbl}


# ============================================================
# Artifacts + stage wrapper
# ============================================================

def write_part_b_artifacts(results: Dict[str, object], cfg: BacktestBConfig, tag: str) -> None:
    if not cfg.write_artifacts:
        return
    out_dir = Path(cfg.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = results.get("meta", {})
    (out_dir / f"part_b__{tag}__meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    pol = results.get("policy_table")
    if isinstance(pol, pd.DataFrame) and len(pol):
        pol.to_csv(out_dir / f"part_b__{tag}__policy.csv", index=False)

    joint = results.get("joint_table")
    if isinstance(joint, pd.DataFrame) and len(joint):
        joint.to_csv(out_dir / f"part_b__{tag}__joint_regimes.csv", index=False)


def stage5_part_b(stage4_df: pd.DataFrame, cfg: Optional[BacktestBConfig] = None) -> pd.DataFrame:
    """
    Pipeline hook: call after stage4.

    Produces two universes:
      1) earnings_day: validates scores against realized reaction labels (your Part A style)
      2) pre_earnings (earnings_window by default): evaluates proximity & positioning BEFORE earnings

    Writes artifacts:
      backtesting/artifacts/part_b__earnings_day__*.csv
      backtesting/artifacts/part_b__pre_earnings__*.csv
    """
    cfg = cfg or BacktestBConfig()

    # --- Universe 1: earnings day (event rows)
    day_df = _filter_rows(stage4_df, cfg.earnings_day_flag_col)

    pol_day = policy_backtests(
        df_events=day_df,
        cfg=cfg,
        score_cols=cfg.policy_score_cols_day,
        universe_name="earnings_day",
    )
    joint_day = joint_regime_tests(day_df, cfg=cfg, universe_name="earnings_day")

    # Merge for convenience
    day_results = {
        "meta": {**pol_day["meta"], **{k: v for k, v in joint_day["meta"].items() if k not in pol_day["meta"]}},
        "policy_table": pol_day["policy_table"],
        "joint_table": joint_day["joint_table"],
    }
    write_part_b_artifacts(day_results, cfg, tag="earnings_day")

    # --- Universe 2: pre-earnings (positioning rows)
    pre_df = _filter_rows(stage4_df, cfg.pre_earnings_flag_col)

    # NOTE:
    # Labels are still derived from earnings-day outcomes, so your dataset must carry them
    # on the pre-earnings rows too (or be joinable by (stock, earnings_date)).
    # If labels are only on earnings-day rows, this will yield near-empty / missing labels.
    pol_pre = policy_backtests(
        df_events=pre_df,
        cfg=cfg,
        score_cols=cfg.policy_score_cols_pre,
        universe_name="pre_earnings",
    )
    joint_pre = joint_regime_tests(pre_df, cfg=cfg, universe_name="pre_earnings")

    pre_results = {
        "meta": {**pol_pre["meta"], **{k: v for k, v in joint_pre["meta"].items() if k not in pol_pre["meta"]}},
        "policy_table": pol_pre["policy_table"],
        "joint_table": joint_pre["joint_table"],
    }
    write_part_b_artifacts(pre_results, cfg, tag="pre_earnings")

    # --- Console summary (compact, PM-friendly)
    print("\n[Stage5 Part B] Policy + regime summary")

    def _print_policy_head(pol_tbl: pd.DataFrame, title: str):
        print(f"\n{title}")
        if pol_tbl is None or not isinstance(pol_tbl, pd.DataFrame) or len(pol_tbl) == 0:
            print("  (no rows)")
            return
        # show best quantile rows for extreme capture vs FN tradeoff
        cols = [
            "universe", "score_col", "q_threshold", "n_flag",
            "p_extreme_flag", "lift_extreme_flag", "capture_extreme", "fn_rate_extreme",
            "abs_p95_all", "abs_p95_hold"
        ]
        show = pol_tbl[cols].sort_values(["score_col", "q_threshold"])
        print(show.to_string(index=False))

    _print_policy_head(day_results.get("policy_table"), "Earnings-day policies")
    _print_policy_head(pre_results.get("policy_table"), "Pre-earnings policies")

    # Joint regimes: show only high-high cell (a_high=1,b_high=1)
    def _print_joint_hh(joint_tbl: pd.DataFrame, title: str):
        print(f"\n{title}")
        if joint_tbl is None or not isinstance(joint_tbl, pd.DataFrame) or len(joint_tbl) == 0:
            print("  (no rows)")
            return
        hh = joint_tbl[(joint_tbl["a_high"] == 1) & (joint_tbl["b_high"] == 1)].copy()
        if len(hh) == 0:
            print("  (no high-high cells)")
            return
        cols = ["universe", "score_a", "score_b", "high_q", "n", "p_extreme", "lift_extreme", "abs_p95", "small_n_flag"]
        print(hh[cols].sort_values(["lift_extreme"], ascending=False).to_string(index=False))

    _print_joint_hh(day_results.get("joint_table"), "Earnings-day joint regimes (high-high)")
    _print_joint_hh(pre_results.get("joint_table"), "Pre-earnings joint regimes (high-high)")

    return stage4_df
