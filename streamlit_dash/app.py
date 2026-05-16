# app.py
import streamlit as st, sys, pandas as pd
from datetime import timedelta
# Streamlit page configuration
st.set_page_config(
    page_title="Breakwater",
    layout="wide"
)

from pathlib import Path
# Add project root (parent of "streamlit") to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# CSV with the dashboard output
# put latest_earnings_df.csv in the repo root (same level as /streamlit)
CSV_PATH = ROOT / "streamlit_df.csv"  # change if you keep it elsewhere
from pipeline.pipeline import run_pipeline
@st.cache_data(show_spinner="Loading parquet…")
def get_full_df() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "output" / "full_df.parquet")
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

@st.cache_data(show_spinner="Loading dashboard data…")
def get_dashboard_df(use_cached_eps: bool = True) -> pd.DataFrame:
    """
    Calls your engine and returns the final dashboard dataframe.

    Assumes run_pipeline(...) returns a DataFrame with at least:
        Date, Stock, risk_level, risk_score, hist_xtreme_prob, base_xtreme_prob, risk_lift
    """
    df = pd.read_csv(CSV_PATH)
    #df = run_pipeline()

    # Sanity check for expected columns
    expected_cols = [
        "stock",
        "sector",
        "sub_sector",
        "earnings_date",
        "is_large_reaction",
        "is_extreme_reaction",
        "hist_extreme_prob",
        "global_hist_prob",
        "current_lift_vs_baseline",
        "current_lift_vs_same_bucket_global",
        "extreme_count",
        "risk_level",
        "risk_score",
        "base_extreme_prob",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.warning(f"Missing expected columns in CSV: {missing}")

    if "earnings_date" in df.columns:
        df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")

    numeric_cols = [
        "risk_score",
        "hist_extreme_prob",
        "global_hist_prob",
        "current_lift_vs_baseline",
        "current_lift_vs_same_bucket_global",
        "base_extreme_prob",
        "extreme_count",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    if "stock" in df.columns:
        stocks = sorted(df["stock"].dropna().unique())
        stock_choice = st.sidebar.selectbox(
            "Stock",
            options=["(All)"] + stocks,
        )
        if stock_choice != "(All)":
            df = df[df["stock"] == stock_choice]

    if "sector" in df.columns:
        sectors = sorted(df["sector"].dropna().unique())
        selected_sectors = st.sidebar.multiselect(
            "Sector",
            options=sectors,
            default=sectors,
        )
        if selected_sectors:
            df = df[df["sector"].isin(selected_sectors)]

    if "risk_level" in df.columns:
        risk_levels = sorted(df["risk_level"].dropna().unique())
        selected_risks = st.sidebar.multiselect(
            "Risk level",
            options=risk_levels,
            default=risk_levels,
        )
        if selected_risks:
            df = df[df["risk_level"].isin(selected_risks)]

    if "risk_score" in df.columns and not df["risk_score"].isna().all():
        min_rs = int(df["risk_score"].min())
        max_rs = int(df["risk_score"].max())

        lo, hi = st.sidebar.slider(
            "Risk score range",
            min_value=min_rs,
            max_value=max_rs,
            value=(min_rs, max_rs),
        )

        df = df[(df["risk_score"] >= lo) & (df["risk_score"] <= hi)]

    if "is_extreme_reaction" in df.columns:
        only_extreme = st.sidebar.checkbox("Only actual extreme reactions", value=False)
        if only_extreme:
            df = df[df["is_extreme_reaction"] == 1]

    if "is_large_reaction" in df.columns:
        only_large = st.sidebar.checkbox("Only actual large reactions", value=False)
        if only_large:
            df = df[df["is_large_reaction"] == 1]

    return df

def main():
    st.title("Breakwater - Earnings Risk Dashboard")

    with st.sidebar:
        st.markdown("### Data options")
        if st.button("Reload CSV from disk"):
            get_dashboard_df.clear()

    raw_df = get_dashboard_df()
    df = sidebar_filters(raw_df.copy())

    if df.empty:
        st.warning("No rows match the current filters.")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Earnings events", len(df))

    with col2:
        if "stock" in df.columns:
            st.metric("Unique stocks", df["stock"].nunique())

    with col3:
        if "risk_score" in df.columns:
            st.metric("Avg risk score", f"{df['risk_score'].mean():.1f}")

    with col4:
        if "is_extreme_reaction" in df.columns:
            st.metric("Extreme reactions", int(df["is_extreme_reaction"].sum()))

    tab_overview, tab_buckets, tab_stock, tab_calendar = st.tabs(
        ["Overview", "Bucket stats", "Stock drill-down", "Weekly Calendar"]
    )

    with tab_overview:
        st.subheader("Filtered earnings events")

        cols_to_show = [
            c for c in [
                "earnings_date",
                "stock",
                "sector",
                "sub_sector",
                "risk_level",
                "risk_score",
                "hist_extreme_prob",
                "base_extreme_prob",
                "current_lift_vs_baseline",
                "current_lift_vs_same_bucket_global",
                "is_large_reaction",
                "is_extreme_reaction",
            ]
            if c in df.columns
        ]

        df_display = df.sort_values("earnings_date", ascending=False)

        st.dataframe(
            df_display[cols_to_show],
            use_container_width=True,
            column_config={
                "earnings_date": st.column_config.DateColumn("Earnings date", format="DD/MM/YYYY"),
                "hist_extreme_prob": st.column_config.NumberColumn("Hist extreme prob", format="%.3f"),
                "base_extreme_prob": st.column_config.NumberColumn("Base extreme prob", format="%.3f"),
                "current_lift_vs_baseline": st.column_config.NumberColumn("Lift vs baseline", format="%.2f"),
                "current_lift_vs_same_bucket_global": st.column_config.NumberColumn("Lift vs same bucket", format="%.2f"),
                "risk_score": st.column_config.NumberColumn("Risk score", format="%.0f"),
            }
        )

        if "risk_level" in df.columns:
            st.markdown("#### Count of events by risk level")
            risk_counts = df["risk_level"].value_counts().sort_index()
            st.bar_chart(risk_counts)

    with tab_buckets:
        st.subheader("Risk bucket statistics")

        bucket_cols = [
            "risk_level",
            "hist_extreme_prob",
            "global_hist_prob",
            "current_lift_vs_baseline",
            "current_lift_vs_same_bucket_global",
            "extreme_count",
        ]

        bucket_cols = [c for c in bucket_cols if c in df.columns]

        bucket_df = (
            df[bucket_cols]
            .drop_duplicates()
            .sort_values("hist_extreme_prob", ascending=False)
        )

        st.dataframe(bucket_df, use_container_width=True)

    with tab_stock:
        st.subheader("Single-stock history")

        if "stock" not in df.columns:
            st.info("stock column not found.")
            return

        stocks = sorted(df["stock"].dropna().unique())
        selected_stock = st.selectbox("Choose stock", options=stocks)

        stock_df = df[df["stock"] == selected_stock].copy()

        if "earnings_date" in stock_df.columns:
            stock_df = stock_df.sort_values("earnings_date")

        cols = [
            c for c in [
                "earnings_date",
                "stock",
                "risk_level",
                "risk_score",
                "hist_extreme_prob",
                "base_extreme_prob",
                "current_lift_vs_baseline",
                "current_lift_vs_same_bucket_global",
                "is_large_reaction",
                "is_extreme_reaction",
            ]
            if c in stock_df.columns
        ]

        st.dataframe(stock_df[cols], use_container_width=True)

        if {"earnings_date", "risk_score"}.issubset(stock_df.columns):
            chart_df = stock_df.set_index("earnings_date")["risk_score"]
            st.line_chart(chart_df)
            
    with tab_calendar:
        st.subheader("Earnings Risk Calendar")

        full_df = get_full_df()
        earn = full_df[full_df["is_earnings_day"] == 1].copy()

        latest_date = earn["earnings_date"].max()
        default_start = latest_date - timedelta(days=7)
        default_end   = latest_date + timedelta(days=7)

        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("From", value=default_start.date())
        with c2:
            end_date = st.date_input("To",   value=default_end.date())

        window = earn[
            (earn["earnings_date"] >= pd.Timestamp(start_date)) &
            (earn["earnings_date"] <= pd.Timestamp(end_date))
        ].copy()

        if window.empty:
            st.info("No earnings events in the selected window.")
        else:
            # Fragility thresholds from full history
            frag_elevated_thr  = earn["momentum_fragility_score"].quantile(0.75)
            frag_stretched_thr = earn["momentum_fragility_score"].quantile(0.90)

            def frag_label(s):
                if pd.isna(s): return ""
                if s >= frag_stretched_thr: return "Stretched"
                if s >= frag_elevated_thr:  return "Elevated"
                return ""

            window["positioning"] = window["momentum_fragility_score"].apply(frag_label)

            display = window[[
                "earnings_date", "stock", "sector", "sub_sector",
                "earnings_explosiveness_score", "earnings_explosiveness_bucket",
                "momentum_fragility_score", "positioning",
            ]].rename(columns={
                "earnings_date":                "Date",
                "stock":                        "Ticker",
                "sector":                       "Sector",
                "sub_sector":                   "Sub-Sector",
                "earnings_explosiveness_score": "Risk Score",
                "earnings_explosiveness_bucket":"Risk Level",
                "momentum_fragility_score":     "Fragility Score",
                "positioning":                  "Positioning",
            }).sort_values(["Date", "Risk Score"], ascending=[True, False])

            # Summary KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Events", len(display))
            k2.metric("Elevated / Stretched positioning",
                      int((display["Positioning"] != "").sum()))
            k3.metric("Avg Risk Score",
                      f"{display['Risk Score'].mean():.1f}")
            sector_top = window["sector"].value_counts().index[0] if not window.empty else "—"
            k4.metric("Most Active Sector", sector_top)

            st.dataframe(
                display,
                use_container_width=True,
                column_config={
                    "Date":         st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                    "Risk Score":   st.column_config.NumberColumn("Risk Score", format="%.0f"),
                    "Fragility Score": st.column_config.NumberColumn("Fragility", format="%.1f"),
                },
            )

            with st.sidebar:
                st.markdown("---")
                if st.button("Export calendar HTML"):
                    sys.path.insert(0, str(ROOT))
                    from report.calendar_builder import generate_calendar
                    generate_calendar(full_df, reference_date=start_date,
                                      window_days=(end_date - start_date).days)
                    st.success("Written to output/weekly_calendar.html")

if __name__ == "__main__":
    main()