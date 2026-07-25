# ==========================================================
# PROJECT: Machine Learning based Buyer Segmentation and
#          Investment Profiling for Real Estate Market Intelligence
# FILE: app.py
# Purpose: Streamlit Dashboard (Live Analytics) - PRO Version
# Author: Satyam Kumar Singh | Parcl Co. Limited | Unified Mentor Project
#
# Requirements (pip install):
#   streamlit pandas numpy plotly scikit-learn openpyxl
#   folium streamlit-folium fpdf2 kaleido
# (The app degrades gracefully if optional libs are missing.)
# ==========================================================

import io
import base64
import warnings
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# Optional dependencies - guarded imports so the app never
# hard-crashes if a package is missing in the environment.
# ----------------------------------------------------------
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import statsmodels.api as _sm  # noqa: F401  (needed by plotly's trendline="ols")
    STATSMODELS_OK = True
except Exception:
    STATSMODELS_OK = False

try:
    import folium
    from folium.plugins import MarkerCluster
    from streamlit_folium import st_folium
    FOLIUM_OK = True
except Exception:
    FOLIUM_OK = False

try:
    from fpdf import FPDF
    FPDF_OK = True
except Exception:
    FPDF_OK = False


# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Parcl Buyer Segmentation Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# SESSION STATE DEFAULTS
# ==========================================================
defaults = {
    "theme": "Light",
    "map_click_country": None,
    "last_refresh": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ==========================================================
# CUSTOM CSS (theme aware)
# ==========================================================
def inject_css(theme: str):
    if theme == "Dark":
        bg, card_bg, text, accent, sub = "#0e1117", "#1c2030", "#f5f5f5", "#7c9cff", "#a6acc4"
    else:
        bg, card_bg, text, accent, sub = "#f7f9fc", "#ffffff", "#111827", "#2b5cff", "#5b6478"

    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {bg}; color: {text}; }}
        .kpi-card {{
            background: {card_bg};
            border-radius: 14px;
            padding: 18px 16px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            border: 1px solid rgba(120,120,120,0.12);
        }}
        .kpi-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 10px 22px rgba(0,0,0,0.18);
        }}
        .kpi-label {{ font-size: 13px; color: {sub}; font-weight: 600; text-transform: uppercase; letter-spacing: .04em;}}
        .kpi-value {{ font-size: 26px; font-weight: 800; color: {accent}; margin-top: 4px;}}
        .section-title {{
            font-size: 22px; font-weight: 800; color: {text};
            border-left: 6px solid {accent}; padding-left: 10px; margin: 18px 0 6px 0;
        }}
        .pill {{
            display:inline-block; background: {accent}22; color:{accent};
            padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; margin-right:6px;
        }}
        /* --- theme-aware text color everywhere --- */
        h1, h2, h3, h4, h5, h6, p, li, label, span,
        .stMarkdown, .stMarkdown p, .stMarkdown li,
        [data-testid="stSidebar"] * ,
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
        .stSelectbox label, .stMultiSelect label, .stSlider label,
        .stRadio label, .stTextInput label, .stDateInput label {{
            color: {text} !important;
        }}
        [data-testid="stSidebar"] {{
            background-color: {card_bg};
            border-right: 1px solid rgba(120,120,120,0.15);
        }}
        .stDataFrame, .stTable {{
            color: {text};
        }}
        div[data-testid="stMetric"] {{
            background: {card_bg}; border-radius: 12px; padding: 10px; border: 1px solid rgba(120,120,120,0.12);
        }}
        /* --- fix multiselect / selectbox dropdown boxes --- */
        div[data-baseweb="select"] > div {{
            background-color: {card_bg} !important;
            border-color: rgba(120,120,120,0.35) !important;
        }}
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div {{
            color: {text} !important;
        }}
        /* dropdown ka opened menu (jab click karke options dikhte hain) */
        ul[data-baseweb="menu"] {{
            background-color: {card_bg} !important;
        }}
        li[role="option"] {{
            color: {text} !important;
            background-color: {card_bg} !important;
        }}
        li[role="option"]:hover {{
            background-color: {bg} !important;
        }}
        /* --- fix buttons (Refresh Data, Download, Generate PDF, etc) --- */
        .stButton > button, .stDownloadButton > button {{
            background-color: {accent} !important;
            color: #ffffff !important;
            border: none !important;
            font-weight: 600 !important;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            background-color: {accent} !important;
            opacity: 0.85;
            color: #ffffff !important;
        }}
        .stButton > button p, .stDownloadButton > button p {{
            color: #ffffff !important;
        }}
        /* --- fix metric value truncation (Segment Summary Cards etc) --- */
        [data-testid="stMetricValue"] {{
            font-size: 20px !important;
            white-space: normal !important;
            overflow: visible !important;
            line-height: 1.2 !important;
        }}
        [data-testid="stMetric"] {{
            overflow: visible !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_money(val):
    try:
        val = float(val)
    except (TypeError, ValueError):
        return "N/A"
    if abs(val) >= 1_000_000:
        return f"${val/1_000_000:.2f}M"
    if abs(val) >= 1_000:
        return f"${val/1_000:.1f}K"
    return f"${val:,.0f}"

def kpi_card(label, value, col):
    with col:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div></div>""",
            unsafe_allow_html=True,
        )


# ==========================================================
# HEADER + THEME TOGGLE
# ==========================================================
top_l, top_r = st.columns([5, 1])
with top_l:
    st.title("🏠 Real Estate Buyer Segmentation & Investment Profiling")
    st.markdown(
        "AI-driven buyer intelligence dashboard for **Parcl Co. Limited** | "
        "Enhanced with Predictive Features, Geo-Intelligence & Simulation"
    )
with top_r:
    st.session_state.theme = st.radio("Theme", ["Light", "Dark"], horizontal=True, label_visibility="collapsed",
                                       index=0 if st.session_state.theme == "Light" else 1)

inject_css(st.session_state.theme)


# ==========================================================
# DATA LOADING (cached)
# ==========================================================
@st.cache_data(show_spinner=False)
def load_data(path="final_segmented_data.csv"):
    df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner=False)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add CLV, Purchase_Likelihood and a synthetic transaction date if absent."""
    df = df.copy()

    if "CLV" not in df.columns:
        rng = np.random.default_rng(7)
        df["CLV"] = (
            df.get("sale_price", pd.Series([500000] * len(df)))
            * (df.get("satisfaction_score", pd.Series([7] * len(df))) / 10)
            * rng.uniform(1.5, 3.0, len(df))
        ).round(0)

    if "Purchase_Likelihood" not in df.columns:
        income_series = df.get("income", pd.Series([0] * len(df)))
        df["Purchase_Likelihood"] = (
            df.get("satisfaction_score", pd.Series([0] * len(df))) * 0.4
            + (df.get("Age", pd.Series([40] * len(df))) < 35).astype(int) * 25
            + (income_series > income_series.median()).astype(int) * 25
            + (df.get("loan_applied", pd.Series(["No"] * len(df))).astype(str).str.lower() == "yes").astype(int) * 20
        ).clip(0, 100).round(1)

    if "transaction_date" not in df.columns:
        rng = np.random.default_rng(42)
        start = pd.Timestamp("2023-01-01")
        end = pd.Timestamp("2025-12-31")
        span = (end - start).days
        df["transaction_date"] = [start + pd.Timedelta(days=int(d)) for d in rng.integers(0, span, len(df))]
    else:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

    return df


try:
    with st.spinner("Loading dataset..."):
        raw_df = load_data()
        df = engineer_features(raw_df)
except Exception as e:
    st.error(f"Could not load final_segmented_data.csv. Run the pipeline script first. Error: {e}")
    st.stop()

# Approximate country centroids for map visualizations (illustrative only)
COUNTRY_CENTROIDS = {
    "United States": (39.8, -98.6), "USA": (39.8, -98.6), "US": (39.8, -98.6),
    "United Kingdom": (54.0, -2.0), "UK": (54.0, -2.0),
    "India": (22.0, 79.0), "China": (35.0, 103.0), "Canada": (56.1, -106.3),
    "Australia": (-25.3, 133.8), "Germany": (51.2, 10.4), "France": (46.6, 2.2),
    "UAE": (23.4, 53.8), "United Arab Emirates": (23.4, 53.8), "Singapore": (1.35, 103.8),
    "Japan": (36.2, 138.3), "Brazil": (-14.2, -51.9), "Mexico": (23.6, -102.5),
    "Spain": (40.4, -3.7), "Italy": (41.9, 12.6), "South Africa": (-30.6, 22.9),
    "Saudi Arabia": (23.9, 45.1), "Qatar": (25.3, 51.2), "Switzerland": (46.8, 8.2),
}


def centroid_for(country):
    return COUNTRY_CENTROIDS.get(str(country), (0.0, 0.0))


# ==========================================================
# SIDEBAR - ADVANCED FILTERS
# ==========================================================
st.sidebar.header("🔎 Filters")
st.sidebar.caption(f"Last refresh: {st.session_state.last_refresh}")

if st.sidebar.button("🔄 Refresh Data"):
    load_data.clear()
    engineer_features.clear()
    st.session_state.last_refresh = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.rerun()

st.sidebar.markdown(f"**Total Records in File:** {len(df)}")

# Multi-selects
def ms(col, label):
    if col in df.columns:
        opts = sorted(df[col].dropna().unique().tolist())
        default_val = [st.session_state.map_click_country] if (col == "country" and st.session_state.map_click_country in opts) else []
        return st.sidebar.multiselect(label, opts, default=default_val)
    return []

sel_countries = ms("country", "Country")
sel_regions = ms("region", "Region")
sel_segments = ms("Segment_Name", "Segment")
sel_purpose = st.sidebar.multiselect(
    "Acquisition Purpose",
    sorted(df["acquisition_purpose"].dropna().unique().tolist()) if "acquisition_purpose" in df.columns else [],
    default=[],
)
sel_client_type = st.sidebar.multiselect(
    "Client Type",
    sorted(df["client_type"].dropna().unique().tolist()) if "client_type" in df.columns else [],
    default=[],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Range Filters")


def range_slider(col, label, is_int=False):
    if col in df.columns and df[col].notna().any():
        lo, hi = float(df[col].min()), float(df[col].max())
        if lo == hi:
            return (lo, hi)
        if is_int:
            lo, hi = int(lo), int(hi)
        return st.sidebar.slider(label, lo, hi, (lo, hi))
    return None


age_range = range_slider("Age", "Age", is_int=True)
price_range = range_slider("sale_price", "Sale Price")
income_range = range_slider("income", "Income")
clv_range = range_slider("CLV", "Customer Lifetime Value (CLV)")
likelihood_range = range_slider("Purchase_Likelihood", "Purchase Likelihood")

st.sidebar.markdown("---")
if "transaction_date" in df.columns:
    min_d, max_d = df["transaction_date"].min(), df["transaction_date"].max()
    date_range = st.sidebar.date_input("Transaction Date Range", (min_d.date(), max_d.date()),
                                        min_value=min_d.date(), max_value=max_d.date())
else:
    date_range = None

st.sidebar.markdown("---")
search_term = st.sidebar.text_input("🔍 Search Client ID / Name")

# ----------------------------------------------------------
# APPLY FILTERS
# ----------------------------------------------------------
filtered_df = df.copy()

if sel_countries:
    filtered_df = filtered_df[filtered_df["country"].isin(sel_countries)]
if sel_regions:
    filtered_df = filtered_df[filtered_df["region"].isin(sel_regions)]
if sel_segments:
    filtered_df = filtered_df[filtered_df["Segment_Name"].isin(sel_segments)]
if sel_purpose and "acquisition_purpose" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["acquisition_purpose"].isin(sel_purpose)]
if sel_client_type and "client_type" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["client_type"].isin(sel_client_type)]

if age_range and "Age" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Age"].between(age_range[0], age_range[1])]
if price_range and "sale_price" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["sale_price"].between(price_range[0], price_range[1])]
if income_range and "income" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["income"].between(income_range[0], income_range[1])]
if clv_range:
    filtered_df = filtered_df[filtered_df["CLV"].between(clv_range[0], clv_range[1])]
if likelihood_range:
    filtered_df = filtered_df[filtered_df["Purchase_Likelihood"].between(likelihood_range[0], likelihood_range[1])]

if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    filtered_df = filtered_df[filtered_df["transaction_date"].between(start_d, end_d)]

if search_term:
    mask = pd.Series(False, index=filtered_df.index)
    if "client_id" in filtered_df.columns:
        mask |= filtered_df["client_id"].astype(str).str.contains(search_term, case=False, na=False)
    if "client_name" in filtered_df.columns:
        mask |= filtered_df["client_name"].astype(str).str.contains(search_term, case=False, na=False)
    filtered_df = filtered_df[mask]

st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df)}")

st.sidebar.markdown("---")
st.sidebar.subheader("⬇️ Export")
csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button("Download CSV", csv_bytes, "filtered_buyers.csv", "text/csv")

excel_buf = io.BytesIO()
try:
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        filtered_df.to_excel(writer, index=False, sheet_name="Filtered Data")
    st.sidebar.download_button("Download Excel", excel_buf.getvalue(), "filtered_buyers.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
except Exception:
    st.sidebar.caption("Excel export needs `openpyxl` installed.")

if len(filtered_df) == 0:
    st.warning("No records match the selected filters. Try changing the filters.")
    st.stop()


# ==========================================================
# TOP KPI STRIP (visible above tabs, always)
# ==========================================================
k1, k2, k3, k4, k5 = st.columns(5)
kpi_card("Total Clients", f"{filtered_df['client_id'].nunique():,}" if "client_id" in filtered_df.columns else len(filtered_df), k1)
kpi_card("Total Transactions", f"{len(filtered_df):,}", k2)
kpi_card("Avg Satisfaction", round(filtered_df.get("satisfaction_score", pd.Series([0])).mean(), 2), k3)
kpi_card("Avg Sale Price", f"{filtered_df.get('sale_price', pd.Series([0])).mean():,.0f}", k4)
kpi_card("Avg CLV", f"${filtered_df['CLV'].mean():,.0f}", k5)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# TABS
# ==========================================================
tab_overview, tab_seg, tab_geo, tab_pred, tab_insights, tab_sim = st.tabs(
    ["📊 Overview", "🧩 Segmentation", "🗺️ Geography & Maps", "🔮 Predictive Analytics",
     "📑 Insights & Reports", "🧪 Simulator"]
)

# ==========================================================
# TAB 1: OVERVIEW
# ==========================================================
with tab_overview:
    st.markdown('<div class="section-title">Clustering Algorithms Comparison & Justification</div>', unsafe_allow_html=True)
    try:
        eval_metrics = pd.read_csv("cluster_evaluation_metrics.csv")
        st.dataframe(eval_metrics, width="stretch")
    except FileNotFoundError:
        st.info("cluster_evaluation_metrics.csv not found. Run the pipeline script first.")

    st.markdown(
        """
        **Model Justification:**
        - Compared **K-Means** and **Hierarchical Clustering**.
        - Selected the best model based on **Silhouette Score** (higher is better), **Davies-Bouldin Index**
          (lower is better), and **Calinski-Harabasz Score** (higher is better).
        - **K-Means** was chosen for its scalability and effectiveness in real estate buyer segmentation.
        """
    )

    st.markdown('<div class="section-title">Market Trend & Simple Forecast</div>', unsafe_allow_html=True)
    if "sale_price" in filtered_df.columns:
        monthly = (
            filtered_df.set_index("transaction_date")["sale_price"]
            .resample("MS").sum().reset_index()
        )
        if len(monthly) >= 2:
            x = np.arange(len(monthly))
            coeffs = np.polyfit(x, monthly["sale_price"], 1)
            future_x = np.arange(len(monthly), len(monthly) + 3)
            future_dates = pd.date_range(monthly["transaction_date"].max() + pd.offsets.MonthBegin(1), periods=3, freq="MS")
            forecast_vals = np.polyval(coeffs, future_x)

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=monthly["transaction_date"], y=monthly["sale_price"],
                                            mode="lines+markers", name="Actual Sales"))
            fig_trend.add_trace(go.Scatter(x=future_dates, y=forecast_vals, mode="lines+markers",
                                            name="Forecast (3mo)", line=dict(dash="dash")))
            fig_trend.update_layout(title="Monthly Sales Value with 3-Month Forecast (linear trend)")
            st.plotly_chart(fig_trend, width="stretch")
        else:
            st.info("Not enough time-series data points to build a trend line.")

    st.markdown('<div class="section-title">🏆 Top Buyers Leaderboard</div>', unsafe_allow_html=True)
    cols_for_lb = [c for c in ["client_id", "country", "Segment_Name", "sale_price", "CLV", "Purchase_Likelihood"] if c in filtered_df.columns]
    leaderboard = filtered_df.sort_values("CLV", ascending=False)[cols_for_lb].head(10)
    st.dataframe(leaderboard, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">⚠️ Anomaly Detection</div>', unsafe_allow_html=True)
    num_cols = [c for c in ["Age", "income", "sale_price", "satisfaction_score", "CLV", "Purchase_Likelihood"] if c in filtered_df.columns]
    if SKLEARN_OK and len(num_cols) >= 3 and len(filtered_df) >= 20:
        iso_data = filtered_df[num_cols].fillna(filtered_df[num_cols].median())
        model = IsolationForest(contamination=0.03, random_state=42)
        preds = model.fit_predict(iso_data)
        anomalies = filtered_df.loc[preds == -1]
        st.caption(f"{len(anomalies)} unusual client(s) flagged out of {len(filtered_df)}.")
        show_cols = [c for c in ["client_id", "country", "Segment_Name"] + num_cols if c in anomalies.columns]
        st.dataframe(anomalies[show_cols].head(15), width="stretch")
    else:
        st.info("Anomaly detection needs scikit-learn and enough numeric records (≥20).")

# ==========================================================
# TAB 2: SEGMENTATION
# ==========================================================
with tab_seg:
    st.markdown('<div class="section-title">1. Buyer Segmentation Overview</div>', unsafe_allow_html=True)
    if "Segment_Name" in filtered_df.columns:
        seg_counts = filtered_df["Segment_Name"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(seg_counts, x="Segment", y="Count", color="Segment",
                                    title="Client Count per Segment"), width="stretch")
        with c2:
            st.plotly_chart(px.pie(seg_counts, names="Segment", values="Count",
                                    title="Segment Share (%)"), width="stretch")
    else:
        st.warning("Segment_Name column not found. Run buyer_segmentation_pipeline.py first.")

    st.markdown('<div class="section-title">2. Investor Behavior Dashboard</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if {"sale_price", "Segment_Name"}.issubset(filtered_df.columns):
            st.plotly_chart(px.box(filtered_df, x="Segment_Name", y="sale_price", color="Segment_Name",
                                    title="Sale Price Distribution by Segment"), width="stretch")
    with c2:
        if {"loan_applied", "Segment_Name"}.issubset(filtered_df.columns):
            loan_counts = filtered_df.groupby(["Segment_Name", "loan_applied"]).size().reset_index(name="Count")
            st.plotly_chart(px.bar(loan_counts, x="Segment_Name", y="Count", color="loan_applied", barmode="group",
                                    title="Loan Applied Status by Segment"), width="stretch")

    st.subheader("Predictive Analytics")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(px.box(filtered_df, x="Segment_Name", y="CLV", color="Segment_Name",
                                title="Customer Lifetime Value by Segment"), width="stretch")
    with c4:
        st.plotly_chart(px.histogram(filtered_df, x="Purchase_Likelihood", color="Segment_Name",
                                      title="Purchase Likelihood Score"), width="stretch")

    st.markdown('<div class="section-title">Advanced Visualizations</div>', unsafe_allow_html=True)

    # Sunburst: Segment -> Purpose -> Country
    hier_cols = [c for c in ["Segment_Name", "acquisition_purpose", "country"] if c in filtered_df.columns]
    if len(hier_cols) == 3:
        st.plotly_chart(px.sunburst(filtered_df, path=hier_cols, title="Segment → Purpose → Country Hierarchy"),
                         width="stretch")

    # Treemap for market share
    if len(hier_cols) >= 2:
        st.plotly_chart(px.treemap(filtered_df, path=hier_cols, values="sale_price" if "sale_price" in filtered_df.columns else None,
                                    title="Market Share Treemap"), width="stretch")

    # Parallel coordinates
    pc_cols = [c for c in ["Age", "income", "sale_price", "satisfaction_score", "CLV", "Purchase_Likelihood"] if c in filtered_df.columns]
    if len(pc_cols) >= 3 and "Segment_Name" in filtered_df.columns:
        pc_df = filtered_df[pc_cols + ["Segment_Name"]].copy()
        pc_df["Segment_Code"] = pc_df["Segment_Name"].astype("category").cat.codes
        st.plotly_chart(
            px.parallel_coordinates(pc_df, dimensions=pc_cols, color="Segment_Code",
                                     title="Parallel Coordinates: Multi-Feature Comparison"),
            width="stretch",
        )

    # Animated bubble chart
    if {"Age", "sale_price", "CLV", "Segment_Name"}.issubset(filtered_df.columns):
        bubble_df = filtered_df.copy()
        if "satisfaction_score" in bubble_df.columns:
            bubble_df["satisfaction_bin"] = bubble_df["satisfaction_score"].round(0)
            st.plotly_chart(
                px.scatter(bubble_df, x="Age", y="sale_price", size="CLV", color="Segment_Name",
                           animation_frame="satisfaction_bin", size_max=45,
                           title="Animated Bubble: Age vs Sale Price (size=CLV, animated by Satisfaction)"),
                width="stretch",
            )

# ==========================================================
# TAB 3: GEOGRAPHY & MAPS
# ==========================================================
with tab_geo:
    st.markdown('<div class="section-title">3. Geographic Buyer Analysis</div>', unsafe_allow_html=True)
    if "region" in filtered_df.columns:
        region_agg_dict = {"Client_Count": ("client_id", "nunique")} if "client_id" in filtered_df.columns else {"Client_Count": ("region", "count")}
        if "sale_price" in filtered_df.columns:
            region_agg_dict["Avg_Sale_Price"] = ("sale_price", "mean")
        if "CLV" in filtered_df.columns:
            region_agg_dict["Avg_CLV"] = ("CLV", "mean")
        region_data = filtered_df.groupby("region").agg(**region_agg_dict).reset_index()

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(region_data, x="region", y="Client_Count", color="region",
                                    title="Buyer Count by Region"), width="stretch")
        with c2:
            country_data = filtered_df["country"].value_counts().reset_index()
            country_data.columns = ["Country", "Count"]
            st.plotly_chart(px.bar(country_data, x="Country", y="Count", color="Country",
                                    title="Buyer Count by Country"), width="stretch")

        if "Segment_Name" in filtered_df.columns:
            region_segment = filtered_df.groupby(["region", "Segment_Name"]).size().reset_index(name="Count")
            st.plotly_chart(px.bar(region_segment, x="region", y="Count", color="Segment_Name", barmode="stack",
                                    title="Segment Distribution Across Regions"), width="stretch")

    st.markdown('<div class="section-title">🌍 Geographic Intelligence: Choropleth</div>', unsafe_allow_html=True)
    if "country" in filtered_df.columns:
        geo_metric = st.selectbox("Metric to visualize", [c for c in ["CLV", "sale_price", "Purchase_Likelihood"] if c in filtered_df.columns] + ["Buyer Count"])
        if geo_metric == "Buyer Count":
            geo_agg = filtered_df.groupby("country").size().reset_index(name="Value")
        else:
            geo_agg = filtered_df.groupby("country")[geo_metric].mean().reset_index(name="Value")
        fig_choro = px.choropleth(geo_agg, locations="country", locationmode="country names", color="Value",
                                   color_continuous_scale="Blues", title=f"{geo_metric} by Country")
        st.plotly_chart(fig_choro, width="stretch")

    st.markdown('<div class="section-title">📍 Interactive Buyer Map</div>', unsafe_allow_html=True)
    if FOLIUM_OK and "country" in filtered_df.columns:
        rng = np.random.default_rng(3)
        m = folium.Map(location=[20, 10], zoom_start=2, tiles="CartoDB positron")
        cluster = MarkerCluster().add_to(m)
        sample = filtered_df.sample(min(300, len(filtered_df)), random_state=1)
        for _, row in sample.iterrows():
            lat0, lon0 = centroid_for(row.get("country"))
            lat = lat0 + rng.uniform(-2, 2)
            lon = lon0 + rng.uniform(-2, 2)
            popup_txt = f"Country: {row.get('country')}<br>Segment: {row.get('Segment_Name', 'N/A')}<br>CLV: ${row.get('CLV', 0):,.0f}"
            folium.CircleMarker(
                location=[lat, lon], radius=5, color="#2b5cff", fill=True, fill_opacity=0.7,
                popup=folium.Popup(popup_txt, max_width=200),
            ).add_to(cluster)

        map_state = st_folium(m, width=None, height=500, key="buyer_map")

        if map_state and map_state.get("last_object_clicked_popup"):
            popup = map_state["last_object_clicked_popup"]
            for c in filtered_df["country"].unique():
                if str(c) in popup:
                    st.session_state.map_click_country = c
                    st.info(f"Map click detected → filtering to **{c}** on next filter change (select it in sidebar Country filter, pre-filled).")
                    break
    else:
        st.info("Install `folium` and `streamlit-folium` to enable the interactive click-to-filter map.")

# ==========================================================
# TAB 4: PREDICTIVE ANALYTICS
# ==========================================================
with tab_pred:
    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    num_cols_all = [c for c in ["Age", "income", "sale_price", "satisfaction_score", "CLV", "Purchase_Likelihood"] if c in filtered_df.columns]
    if len(num_cols_all) >= 2:
        corr = filtered_df[num_cols_all].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Feature Correlation Matrix")
        st.plotly_chart(fig_corr, width="stretch")

        pair_a = st.selectbox("Inspect variable pair — X axis", num_cols_all, index=0)
        pair_b = st.selectbox("Inspect variable pair — Y axis", num_cols_all, index=min(1, len(num_cols_all) - 1))
        st.plotly_chart(px.scatter(filtered_df, x=pair_a, y=pair_b, color="Segment_Name" if "Segment_Name" in filtered_df.columns else None,
                                    trendline="ols" if STATSMODELS_OK else None, title=f"{pair_a} vs {pair_b}"),
                         width="stretch")
        if not STATSMODELS_OK:
            st.caption("Install `statsmodels` (`pip install statsmodels`) to overlay a trendline on this scatter plot.")

    st.markdown('<div class="section-title">Cohort Analysis</div>', unsafe_allow_html=True)
    if "Segment_Name" in filtered_df.columns:
        cohort_df = filtered_df.copy()
        cohort_df["Cohort_Month"] = cohort_df["transaction_date"].dt.to_period("M").astype(str)
        cohort_table = pd.crosstab(cohort_df["Cohort_Month"], cohort_df["Segment_Name"])
        if not cohort_table.empty:
            st.plotly_chart(px.imshow(cohort_table, aspect="auto", color_continuous_scale="Blues",
                                       title="Cohort Heatmap: New Clients per Segment per Month",
                                       labels=dict(x="Segment", y="Cohort Month", color="Clients")),
                             width="stretch")

    st.markdown('<div class="section-title">RFM Analysis</div>', unsafe_allow_html=True)
    if {"client_id", "sale_price", "transaction_date"}.issubset(filtered_df.columns):
        snapshot_date = filtered_df["transaction_date"].max() + pd.Timedelta(days=1)
        rfm = filtered_df.groupby("client_id").agg(
            Recency=("transaction_date", lambda x: (snapshot_date - x.max()).days),
            Frequency=("transaction_date", "count"),
            Monetary=("sale_price", "sum"),
        ).reset_index()
        rfm["R_Score"] = pd.qcut(rfm["Recency"], 4, labels=[4, 3, 2, 1], duplicates="drop").astype(int)
        rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4], duplicates="drop").astype(int)
        rfm["M_Score"] = pd.qcut(rfm["Monetary"], 4, labels=[1, 2, 3, 4], duplicates="drop").astype(int)
        rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
        st.dataframe(rfm.sort_values("RFM_Score", ascending=False).head(20), width="stretch")
    else:
        st.info("RFM needs client_id, sale_price and transaction_date columns.")

    st.markdown('<div class="section-title">ROI / Opportunity Score by Segment</div>', unsafe_allow_html=True)
    if "Segment_Name" in filtered_df.columns:
        seg_metrics = filtered_df.groupby("Segment_Name").agg(
            Avg_CLV=("CLV", "mean"),
            Avg_Likelihood=("Purchase_Likelihood", "mean"),
            Avg_Satisfaction=("satisfaction_score", "mean") if "satisfaction_score" in filtered_df.columns else ("CLV", "mean"),
        ).reset_index()

        def norm(s):
            return (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else s * 0

        seg_metrics["Opportunity_Score"] = (
            norm(seg_metrics["Avg_CLV"]) * 0.5
            + norm(seg_metrics["Avg_Likelihood"]) * 0.3
            + norm(seg_metrics["Avg_Satisfaction"]) * 0.2
        ) * 100
        st.plotly_chart(px.bar(seg_metrics.sort_values("Opportunity_Score", ascending=False),
                                x="Segment_Name", y="Opportunity_Score", color="Segment_Name",
                                title="Opportunity Score (0-100) — weighted CLV + Likelihood + Satisfaction"),
                         width="stretch")
        st.caption("Opportunity Score = 0.5×norm(CLV) + 0.3×norm(Purchase Likelihood) + 0.2×norm(Satisfaction)")

# ==========================================================
# TAB 5: INSIGHTS & REPORTS
# ==========================================================
with tab_insights:
    st.markdown('<div class="section-title">Segment Summary Cards</div>', unsafe_allow_html=True)
    if "Segment_Name" in filtered_df.columns:
        segments_list = filtered_df["Segment_Name"].unique()
        cards_per_row = 2
        rows_needed = (len(segments_list) + cards_per_row - 1) // cards_per_row
        seg_idx = 0
        for r in range(rows_needed):
            cols = st.columns(cards_per_row)
            for c in cols:
                if seg_idx >= len(segments_list):
                    break
                seg_name = segments_list[seg_idx]
                seg_data = filtered_df[filtered_df["Segment_Name"] == seg_name]
                with c:
                    st.markdown(f"##### {seg_name}")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Clients", seg_data["client_id"].nunique() if "client_id" in seg_data.columns else len(seg_data))
                    m2.metric("Avg Age", round(seg_data["Age"].mean(), 1) if "Age" in seg_data.columns and len(seg_data) else "N/A")
                    m3.metric("Avg Satisfaction", round(seg_data["satisfaction_score"].mean(), 2) if "satisfaction_score" in seg_data.columns and len(seg_data) else "N/A")
                    m4.metric("Avg CLV", format_money(seg_data["CLV"].mean()) if "CLV" in seg_data.columns and len(seg_data) else "N/A")
                seg_idx += 1

    st.markdown('<div class="section-title">4. Segment Insights Panel</div>', unsafe_allow_html=True)
    insights = None
    if "Segment_Name" in filtered_df.columns:
        insights_agg_dict = {
            "Total_Clients": ("client_id", "nunique") if "client_id" in filtered_df.columns else ("Segment_Name", "count"),
            "Avg_Age": ("Age", "mean"),
            "Avg_Satisfaction": ("satisfaction_score", "mean"),
            "Avg_CLV": ("CLV", "mean"),
            "Avg_Likelihood": ("Purchase_Likelihood", "mean"),
        }
        if "loan_applied" in filtered_df.columns:
            insights_agg_dict["Loan_Applied_Rate"] = ("loan_applied", lambda x: (x.astype(str).str.lower() == "yes").mean())
        if "sale_price" in filtered_df.columns:
            insights_agg_dict["Avg_Sale_Price"] = ("sale_price", "mean")

        insights = filtered_df.groupby("Segment_Name").agg(**insights_agg_dict).reset_index()
        if "Avg_Age" in insights.columns:
            insights["Avg_Age"] = insights["Avg_Age"].round(1)
        if "Avg_Satisfaction" in insights.columns:
            insights["Avg_Satisfaction"] = insights["Avg_Satisfaction"].round(2)
        insights["Avg_CLV"] = insights["Avg_CLV"].round(0)
        insights["Avg_Likelihood"] = insights["Avg_Likelihood"].round(1)
        if "Avg_Sale_Price" in insights.columns:
            insights["Avg_Sale_Price"] = insights["Avg_Sale_Price"].apply(lambda x: f"{x:,.0f}")
        if "Loan_Applied_Rate" in insights.columns:
            insights["Loan_Applied_Rate"] = (insights["Loan_Applied_Rate"] * 100).round(1).astype(str) + "%"

        st.dataframe(insights, width="stretch")

        selected_segment = st.selectbox("View detailed records for segment:", insights["Segment_Name"].unique())
        st.dataframe(filtered_df[filtered_df["Segment_Name"] == selected_segment], width="stretch")

    st.markdown('<div class="section-title">Recommended Buyer Segments</div>', unsafe_allow_html=True)
    segment_cluster_map = {
        "Global Investors": ("C1", "High income, investment purchases"),
        "First-Time Buyers": ("C2", "Younger, loan dependent"),
        "Corporate Buyers": ("C3", "Companies purchasing multiple units"),
        "Luxury Investors": ("C4", "High satisfaction, large investments"),
    }
    if "Segment_Name" in filtered_df.columns:
        rows = []
        for segment_name, (cluster_code, characteristics) in segment_cluster_map.items():
            segment_data = filtered_df[filtered_df["Segment_Name"] == segment_name]
            client_count = segment_data["client_id"].nunique() if "client_id" in segment_data.columns else len(segment_data)
            avg_age = f"{segment_data['Age'].mean():.1f}" if len(segment_data) > 0 and "Age" in segment_data.columns else "N/A"
            avg_satisfaction = f"{segment_data['satisfaction_score'].mean():.2f}" if len(segment_data) > 0 and "satisfaction_score" in segment_data.columns else "N/A"
            avg_price = f"{segment_data['sale_price'].mean():,.0f}" if "sale_price" in segment_data.columns and len(segment_data) > 0 else "N/A"
            rows.append({"Cluster": cluster_code, "Buyer Type": segment_name, "Characteristics": characteristics,
                         "Client Count": client_count, "Avg Age": avg_age, "Avg Satisfaction": avg_satisfaction,
                         "Avg Sale Price": avg_price})
        st.table(pd.DataFrame(rows))

    st.markdown('<div class="section-title">Cluster Evaluation Metrics & Visualizations</div>', unsafe_allow_html=True)
    try:
        eval_metrics2 = pd.read_csv("cluster_evaluation_metrics.csv")
        st.table(eval_metrics2)
    except FileNotFoundError:
        st.info("cluster_evaluation_metrics.csv not found.")

    img_c1, img_c2 = st.columns(2)
    for col, fname, cap in [
        (img_c1, "pca_kmeans_clusters.png", "K-Means Clusters (PCA 2D Projection)"),
        (img_c2, "pca_hierarchical_clusters.png", "Hierarchical Clusters (PCA 2D Projection)"),
    ]:
        with col:
            try:
                st.image(fname, caption=cap, width="stretch")
            except Exception:
                st.info(f"{fname} not found.")
    try:
        st.image("hierarchical_dendrogram.png", caption="Hierarchical Clustering Dendrogram", width="stretch")
    except Exception:
        st.info("hierarchical_dendrogram.png not found.")

    st.markdown('<div class="section-title">📄 Generate Professional PDF Report</div>', unsafe_allow_html=True)
    report_segment = st.selectbox("Select segment for the report", filtered_df["Segment_Name"].unique() if "Segment_Name" in filtered_df.columns else ["All"])
    if st.button("Generate PDF Report"):
        if not FPDF_OK:
            st.error("Install `fpdf2` (`pip install fpdf2`) to enable PDF report generation.")
        else:
            seg_data = filtered_df[filtered_df["Segment_Name"] == report_segment] if "Segment_Name" in filtered_df.columns else filtered_df
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Parcl Buyer Segmentation Report", ln=True)
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 8, f"Segment: {report_segment}", ln=True)
            pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Key Metrics", ln=True)
            pdf.set_font("Helvetica", "", 11)
            metrics_text = [
                f"Total Clients: {seg_data['client_id'].nunique() if 'client_id' in seg_data.columns else len(seg_data)}",
                f"Avg Age: {seg_data['Age'].mean():.1f}" if "Age" in seg_data.columns else "",
                f"Avg Satisfaction: {seg_data['satisfaction_score'].mean():.2f}" if "satisfaction_score" in seg_data.columns else "",
                f"Avg Sale Price: {seg_data['sale_price'].mean():,.0f}" if "sale_price" in seg_data.columns else "",
                f"Avg CLV: ${seg_data['CLV'].mean():,.0f}",
                f"Avg Purchase Likelihood: {seg_data['Purchase_Likelihood'].mean():.1f}",
            ]
            for line in metrics_text:
                if line:
                    pdf.cell(0, 7, line, ln=True)

            pdf_bytes = pdf.output(dest="S")
            if isinstance(pdf_bytes, str):
                pdf_bytes = pdf_bytes.encode("latin-1")
            st.download_button("⬇️ Download PDF Report", data=bytes(pdf_bytes),
                                file_name=f"parcl_report_{report_segment.replace(' ', '_')}.pdf",
                                mime="application/pdf")

# ==========================================================
# TAB 6: WHAT-IF SIMULATOR
# ==========================================================
with tab_sim:
    st.markdown('<div class="section-title">🧪 What-If Investment Simulator</div>', unsafe_allow_html=True)
    st.caption("Adjust the sliders to simulate a hypothetical buyer and preview predicted CLV, "
               "Purchase Likelihood and the most likely segment.")

    s1, s2, s3 = st.columns(3)
    with s1:
        sim_age = st.slider("Age", 18, 80, 35)
        sim_income = st.slider("Income", int(df.get("income", pd.Series([20000])).min()),
                                int(df.get("income", pd.Series([500000])).max()),
                                int(df.get("income", pd.Series([100000])).median()))
    with s2:
        sim_satisfaction = st.slider("Satisfaction Score", 0, 10, 7)
        sim_price = st.slider("Sale Price", int(df.get("sale_price", pd.Series([50000])).min()),
                               int(df.get("sale_price", pd.Series([2000000])).max()),
                               int(df.get("sale_price", pd.Series([300000])).median()))
    with s3:
        sim_loan = st.selectbox("Loan Applied", ["Yes", "No"])
        sim_purpose = st.selectbox("Acquisition Purpose",
                                    sorted(df["acquisition_purpose"].dropna().unique().tolist()) if "acquisition_purpose" in df.columns else ["Investment"])
        sim_client_type = st.selectbox("Client Type",
                                        sorted(df["client_type"].dropna().unique().tolist()) if "client_type" in df.columns else ["Individual"])

    # ---- Predicted CLV (same formula family as the pipeline) ----
    sim_clv = sim_price * (sim_satisfaction / 10) * 2.2  # midpoint of 1.5-3.0 multiplier range
    sim_likelihood = (
        sim_satisfaction * 0.4
        + (1 if sim_age < 35 else 0) * 25
        + (1 if sim_income > df.get("income", pd.Series([sim_income])).median() else 0) * 25
        + (1 if sim_loan == "Yes" else 0) * 20
    )
    sim_likelihood = float(np.clip(sim_likelihood, 0, 100))

    r1, r2 = st.columns(2)
    kpi_card("Predicted CLV", f"${sim_clv:,.0f}", r1)
    kpi_card("Predicted Purchase Likelihood", f"{sim_likelihood:.1f} / 100", r2)

    st.markdown('<div class="section-title">Segment Recommendation Engine</div>', unsafe_allow_html=True)
    numeric_feats = [c for c in ["Age", "income", "sale_price", "satisfaction_score"] if c in df.columns]
    categorical_feats = [c for c in ["client_type", "acquisition_purpose"] if c in df.columns]

    if SKLEARN_OK and "Segment_Name" in df.columns and len(numeric_feats) >= 3:

        # Show how the segments are distributed in the data — if one segment
        # (e.g. Luxury Investors) dominates, the vote will lean toward it
        # for most "typical" slider values. This isn't a bug, it's the data.
        seg_dist = df["Segment_Name"].value_counts(normalize=True).mul(100).round(1)
        with st.expander("ℹ️ Why does the recommendation lean toward one segment?", expanded=True):
            st.write(
                "The recommender now looks at both **numeric traits** (Age, Income, Sale Price, "
                "Satisfaction) and **categorical traits** (Client Type, Acquisition Purpose) of the "
                "7 nearest buyers to your simulated profile. Segments that are mostly separated by "
                "category (e.g. Corporate Buyers = companies) rather than by numbers were previously "
                "invisible to the model — this should now surface them correctly. Current segment "
                "share in the data:"
            )
            st.dataframe(seg_dist.rename("Share (%)"), use_container_width=True)

        @st.cache_resource(show_spinner=False)
        def train_knn(_df, num_feats, cat_feats):
            X_num = _df[num_feats].fillna(_df[num_feats].median())
            scaler = StandardScaler().fit(X_num)
            Xs_num = scaler.transform(X_num)

            cat_dummies = pd.DataFrame(index=_df.index)
            cat_columns_used = []
            if cat_feats:
                cat_dummies = pd.get_dummies(_df[cat_feats].astype(str), prefix=cat_feats)
                cat_columns_used = cat_dummies.columns.tolist()
                # Weight categorical match as strongly as the numeric block combined,
                # so a Corporate Buyer isn't drowned out by 4 numeric dimensions.
                cat_weight = np.sqrt(len(num_feats)) if cat_columns_used else 1.0
                Xs_cat = cat_dummies.values.astype(float) * cat_weight
                Xs = np.hstack([Xs_num, Xs_cat])
            else:
                Xs = Xs_num

            y = _df["Segment_Name"]
            knn = KNeighborsClassifier(n_neighbors=7, weights="distance").fit(Xs, y)
            return scaler, cat_columns_used, knn

        scaler, cat_columns_used, knn = train_knn(df, numeric_feats, categorical_feats)

        sim_row = {"Age": sim_age, "income": sim_income, "sale_price": sim_price, "satisfaction_score": sim_satisfaction}
        sim_vector = pd.DataFrame([sim_row])[numeric_feats]
        sim_scaled_num = scaler.transform(sim_vector)

        if cat_columns_used:
            sim_cat_raw = {}
            if "client_type" in categorical_feats:
                sim_cat_raw["client_type"] = sim_client_type
            if "acquisition_purpose" in categorical_feats:
                sim_cat_raw["acquisition_purpose"] = sim_purpose
            sim_cat_df = pd.DataFrame([sim_cat_raw]).astype(str)
            sim_dummies = pd.get_dummies(sim_cat_df, prefix=categorical_feats)
            sim_dummies = sim_dummies.reindex(columns=cat_columns_used, fill_value=0)
            cat_weight = np.sqrt(len(numeric_feats))
            sim_scaled = np.hstack([sim_scaled_num, sim_dummies.values.astype(float) * cat_weight])
        else:
            sim_scaled = sim_scaled_num

        predicted_segment = knn.predict(sim_scaled)[0]
        proba = knn.predict_proba(sim_scaled)[0]
        classes = knn.classes_

        st.success(f"**Recommended Segment: {predicted_segment}**")

        proba_df = pd.DataFrame({"Segment": classes, "Confidence": proba}).sort_values("Confidence", ascending=False)
        st.plotly_chart(px.bar(proba_df, x="Segment", y="Confidence", color="Segment",
                                title="Segment Match Confidence (K-Nearest-Neighbors, k=7, numeric + categorical)"),
                         width="stretch")

        # Reasoning: compare simulated buyer to segment averages (numeric feats only)
        feature_cols = numeric_feats
        seg_avg = df.groupby("Segment_Name")[feature_cols].mean()
        if predicted_segment in seg_avg.index:
            comp = seg_avg.loc[predicted_segment]
            reasoning = []
            for feat in feature_cols:
                sim_val = sim_vector[feat].iloc[0]
                seg_val = comp[feat]
                diff = sim_val - seg_val
                # Treat anything within 2% of the segment average as "similar"
                # instead of mislabeling near-identical numbers as above/below.
                tolerance = max(abs(seg_val) * 0.02, 0.05)
                if abs(diff) <= tolerance:
                    relation = "about the same as"
                else:
                    relation = "above" if diff > 0 else "below"
                reasoning.append(f"- **{feat}** is {relation} the *{predicted_segment}* segment average "
                                  f"({sim_val:,.0f} vs {seg_val:,.0f}).")
            st.markdown("**Why this segment?**\n" + "\n".join(reasoning))
    else:
        st.info("Segment recommendation needs scikit-learn plus Age, income, sale_price and satisfaction_score columns.")

st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Dashboard Built by Satyam Kumar Singh | Parcl Co. Limited | "
    "Unified Mentor Project</p>",
    unsafe_allow_html=True,
)
