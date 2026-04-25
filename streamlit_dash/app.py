# app.py
import streamlit as st, sys, pandas as pd
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

    tab_overview, tab_buckets, tab_stock = st.tabs(
        ["Overview", "Bucket stats", "Stock drill-down"]
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
            
# def main():
#     st.title("Breakwater - Earnings Risk & Alerts Dashboard")

#     with st.sidebar:
#         st.markdown("### Data options")
#         if st.button("Reload CSV from disk"):
#             # clear cache and reload on next get_dashboard_df() call
#             get_dashboard_df.clear()

#     raw_df = get_dashboard_df()
#     df = raw_df.copy()

#     # Apply sidebar filters
#     df = sidebar_filters(df)

#     if df.empty:
#         st.warning("No rows match the current filters.")
#         return

#     # High-level KPIs
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Earnings events", len(df))
#     with col2:
#         if "Stock" in df.columns:
#             st.metric("Unique stocks", df["Stock"].nunique())
#     with col3:
#         if "risk_score" in df.columns:
#             # naive threshold: 4+ considered high risk
#             high_risk = (df["risk_score"] >= 4).sum()
#             st.metric("High-risk events (Score ≥ 4)", int(high_risk))
#     with col4:
#         if "any_alert" in df.columns:
#             st.metric("Rows with alerts", int(df["any_alert"].sum()))

#     # Tabs: Overview / Alerts / Stock detail
#     tab_overview, tab_alerts, tab_stock = st.tabs(
#         ["Overview", "Risk Alerts", "Stock drill-down"]
#     )

#     # -------- Overview tab --------
#     with tab_overview:
#         st.subheader("Filtered earnings events")

#         cols_to_show = [
#             c
#             for c in [
#                 "Date",
#                 "Stock",
#                 "risk_level",
#                 "risk_score",
#                 "hist_xtreme_prob",
#                 "base_xtreme_prob",
#                 "risk_lift",
#                 # "Recommendation",
#                 # "Excessive Move",
#                 # "No Reaction",
#                 # "Reaction Divergence",
#                 # "Muted Response",
#                 # "Extreme Volatility",
#                 # "Divergence Alert",
#             ]
#             if c in df.columns
#         ]

#         if "Date" in df.columns:
#             df_display = df.sort_values("Date", ascending=False)
#         else:
#             df_display = df

#         st.dataframe(
#             df_display[cols_to_show],
#             column_config={
#                 "Date": st.column_config.DateColumn(format="DD/MM/YYYY")
#             }
#         )
#         # st.dataframe(df_display[cols_to_show])

#         # Simple aggregate: count of events by risk_score
#         if "risk_score" in df.columns:
#             st.markdown("#### Count of events by risk_score")
#             agg = (
#                 df.groupby("risk_score")["Stock"]
#                 .count()
#                 .rename("count")
#                 .reset_index()
#                 .sort_values("risk_score")
#             )
#             chart_df = agg.set_index("risk_score")["count"]
#             st.bar_chart(chart_df)

#     # -------- Risk Alerts tab --------
#     with tab_alerts:
#         st.subheader("Flagged risk cases")

#         if "any_alert" not in df.columns or not df["any_alert"].any():
#             st.info("No rows with alert flags in the filtered data.")
#         else:
#             alerts = df[df["any_alert"]].copy()
#             if "Date" in alerts.columns:
#                 alerts = alerts.sort_values("Date", ascending=False)

#             alert_cols = [
#                 c
#                 for c in [
#                     "Date",
#                     "Stock",
#                     "risk_level",
#                     "risk_score",
#                     "hist_xtreme_prob",
#                     "base_xtreme_prob",
#                     "risk_lift",
#                     # "Recommendation",
#                     # "Excessive Move",
#                     # "No Reaction",
#                     # "Reaction Divergence",
#                     # "Muted Response",
#                     # "Extreme Volatility",
#                     # "Divergence Alert",
#                 ]
#                 if c in alerts.columns
#             ]

#             st.dataframe(alerts[alert_cols])

#     # -------- Stock drill-down tab --------
#     with tab_stock:
#         st.subheader("Single-stock history")

#         if "Stock" not in df.columns:
#             st.info("Stock column not found.")
#         else:
#             stocks = sorted(df["Stock"].dropna().unique())
#             selected_stock = st.selectbox("Choose stock", options=stocks)

#             stock_df = df[df["Stock"] == selected_stock].copy()
#             if "Date" in stock_df.columns:
#                 stock_df = stock_df.sort_values("Date")

#             cols = [
#                 c
#                 for c in [
#                     "Date",
#                     "Stock",
#                     "risk_level",
#                     "risk_score",
#                     "hist_xtreme_prob",
#                     "base_xtreme_prob",
#                     "risk_lift",
#                     # "Recommendation",
#                     # "Excessive Move",
#                     # "No Reaction",
#                     # "Reaction Divergence",
#                     # "Muted Response",
#                     # "Extreme Volatility",
#                     # "Divergence Alert",
#                 ]
#                 if c in stock_df.columns
#             ]
#             st.dataframe(stock_df[cols])

#             # Quick line chart: risk_score over time
#             if {"Date", "risk_score"}.issubset(stock_df.columns):
#                 chart_df = stock_df.set_index("Date")["risk_score"]
#                 st.line_chart(chart_df)

if __name__ == "__main__":
    main()