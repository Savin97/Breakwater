# app.py
import streamlit as st, sys, pandas as pd
# Streamlit page configuration
st.set_page_config(
    page_title="Dashboard",
    layout="wide"
)

from pathlib import Path
# Add project root (parent of "streamlit") to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# CSV with the dashboard output
# put latest_earnings_df.csv in the repo root (same level as /streamlit)
CSV_PATH = ROOT / "partial_df.csv"  # change if you keep it elsewhere
from pipeline.pipeline import run_pipeline
@st.cache_data(show_spinner="Running Pipeline…")
def get_dashboard_df(use_cached_eps: bool = True) -> pd.DataFrame:
    """
    Calls your engine and returns the final dashboard dataframe.

    Assumes run_pipeline(...) returns a DataFrame with at least:
        Date, Stock, Risk Level, Risk Score, hist_xtreme_prob, base_xtreme_prob, risk_lift,
        Excessive Move, Reaction Divergence, Muted Response, 
        Extreme Volatility, Divergence Alert, Recommendation,
    """
    df = pd.read_csv("partial_df.csv")
    #df = run_pipeline()

    # Sanity check for expected columns
    expected_cols = [
        "Date",
        "Stock",
        "Risk Level",
        "Risk Score",
        "hist_xtreme_prob",
        "base_xtreme_prob",
        "risk_lift",
        # "Recommendation",
        # "Excessive Move",
        # "Reaction Divergence",
        # "Muted Response",
        # "Extreme Volatility",
        # "Divergence Alert",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.warning(f"Missing expected columns in CSV: {missing}")

    # Convenience: pre-compute any_alert flag
    df["any_alert"] = False
    if "Reaction Divergence" in df.columns:
        df["any_alert"] = df["any_alert"] | df["Reaction Divergence"].fillna(False)
    if "Muted Response" in df.columns:
        df["any_alert"] = df["any_alert"] | df["Muted Response"].fillna(False)
    if "Extreme Volatility" in df.columns:
        df["any_alert"] = df["any_alert"] | (df["Extreme Volatility"].fillna(0) != 0)
    if "No Reaction" in df.columns:
        df["any_alert"] = df["any_alert"] | df["No Reaction"].notna()
    if "Excessive Move" in df.columns:
        # flag anything that is not the plain "No - Within normal range." text
        df["any_alert"] = df["any_alert"] | ~df["Excessive Move"].fillna("").str.contains(
            "No - Within normal range.", na=False
        )
    if "Divergence Alert" in df.columns:
        df["any_alert"] = df["any_alert"] | df["Divergence Alert"].notna()

    return df

def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    # Stock filter
    if "Stock" in df.columns:
        stocks = sorted(df["Stock"].dropna().unique())
        stock_choice = st.sidebar.selectbox(
            "Stock (optional)",
            options=["(All)"] + stocks,
        )
        if stock_choice != "(All)":
            df = df[df["Stock"] == stock_choice]

    # Risk score range
    if "Risk Score" in df.columns and not df["Risk Score"].isna().all():
        min_rs = int(df["Risk Score"].min())
        max_rs = int(df["Risk Score"].max())
        lo, hi = st.sidebar.slider(
            "Risk Score range",
            min_value=min_rs,
            max_value=max_rs,
            value=(min_rs, max_rs),
        )
        df = df[(df["Risk Score"] >= lo) & (df["Risk Score"] <= hi)]

    # Recommendation filter
    if "Recommendation" in df.columns:
        recs = sorted(df["Recommendation"].dropna().unique())
        selected_recs = st.sidebar.multiselect(
            "Recommendations",
            options=recs,
            default=recs,
        )
        if selected_recs:
            df = df[df["Recommendation"].isin(selected_recs)]

    # Only rows with any alert?
    if "any_alert" in df.columns:
        only_alerts = st.sidebar.checkbox(
            "Only rows with any risk/alert flag",
            value=False,
        )
        if only_alerts:
            df = df[df["any_alert"]]

    return df


def main():
    st.title("Breakwater - Earnings Risk & Alerts Dashboard")

    with st.sidebar:
        st.markdown("### Data options")
        if st.button("Reload CSV from disk"):
            # clear cache and reload on next get_dashboard_df() call
            get_dashboard_df.clear()

    raw_df = get_dashboard_df()
    df = raw_df.copy()

    # Apply sidebar filters
    df = sidebar_filters(df)

    if df.empty:
        st.warning("No rows match the current filters.")
        return

    # High-level KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Earnings events", len(df))
    with col2:
        if "Stock" in df.columns:
            st.metric("Unique stocks", df["Stock"].nunique())
    with col3:
        if "Risk Score" in df.columns:
            # naive threshold: 4+ considered high risk
            high_risk = (df["Risk Score"] >= 4).sum()
            st.metric("High-risk events (Score ≥ 4)", int(high_risk))
    with col4:
        if "any_alert" in df.columns:
            st.metric("Rows with alerts", int(df["any_alert"].sum()))

    # Tabs: Overview / Alerts / Stock detail
    tab_overview, tab_alerts, tab_stock = st.tabs(
        ["Overview", "Risk Alerts", "Stock drill-down"]
    )

    # -------- Overview tab --------
    with tab_overview:
        st.subheader("Filtered earnings events")

        cols_to_show = [
            c
            for c in [
                "Date",
                "Stock",
                "Risk Level",
                "Risk Score",
                "hist_xtreme_prob",
                "base_xtreme_prob",
                "risk_lift",
                # "Recommendation",
                # "Excessive Move",
                # "No Reaction",
                # "Reaction Divergence",
                # "Muted Response",
                # "Extreme Volatility",
                # "Divergence Alert",
            ]
            if c in df.columns
        ]

        if "Date" in df.columns:
            df_display = df.sort_values("Date", ascending=False)
        else:
            df_display = df

        st.dataframe(
            df_display[cols_to_show],
            column_config={
                "Date": st.column_config.DateColumn(format="DD/MM/YYYY")
            }
        )
        # st.dataframe(df_display[cols_to_show])

        # Simple aggregate: count of events by risk score
        if "Risk Score" in df.columns:
            st.markdown("#### Count of events by Risk Score")
            agg = (
                df.groupby("Risk Score")["Stock"]
                .count()
                .rename("count")
                .reset_index()
                .sort_values("Risk Score")
            )
            chart_df = agg.set_index("Risk Score")["count"]
            st.bar_chart(chart_df)

    # -------- Risk Alerts tab --------
    with tab_alerts:
        st.subheader("Flagged risk cases")

        if "any_alert" not in df.columns or not df["any_alert"].any():
            st.info("No rows with alert flags in the filtered data.")
        else:
            alerts = df[df["any_alert"]].copy()
            if "Date" in alerts.columns:
                alerts = alerts.sort_values("Date", ascending=False)

            alert_cols = [
                c
                for c in [
                    "Date",
                    "Stock",
                    "Risk Level",
                    "Risk Score",
                    "hist_xtreme_prob",
                    "base_xtreme_prob",
                    "risk_lift",
                    # "Recommendation",
                    # "Excessive Move",
                    # "No Reaction",
                    # "Reaction Divergence",
                    # "Muted Response",
                    # "Extreme Volatility",
                    # "Divergence Alert",
                ]
                if c in alerts.columns
            ]

            st.dataframe(alerts[alert_cols])

    # -------- Stock drill-down tab --------
    with tab_stock:
        st.subheader("Single-stock history")

        if "Stock" not in df.columns:
            st.info("Stock column not found.")
        else:
            stocks = sorted(df["Stock"].dropna().unique())
            selected_stock = st.selectbox("Choose stock", options=stocks)

            stock_df = df[df["Stock"] == selected_stock].copy()
            if "Date" in stock_df.columns:
                stock_df = stock_df.sort_values("Date")

            cols = [
                c
                for c in [
                    "Date",
                    "Stock",
                    "Risk Level",
                    "Risk Score",
                    "hist_xtreme_prob",
                    "base_xtreme_prob",
                    "risk_lift",
                    # "Recommendation",
                    # "Excessive Move",
                    # "No Reaction",
                    # "Reaction Divergence",
                    # "Muted Response",
                    # "Extreme Volatility",
                    # "Divergence Alert",
                ]
                if c in stock_df.columns
            ]
            st.dataframe(stock_df[cols])

            # Quick line chart: Risk Score over time
            if {"Date", "Risk Score"}.issubset(stock_df.columns):
                chart_df = stock_df.set_index("Date")["Risk Score"]
                st.line_chart(chart_df)

if __name__ == "__main__":
    main()