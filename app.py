# ==========================================================
# PROJECT: Machine Learning based Buyer Segmentation and
#          Investment Profiling for Real Estate Market Intelligence
# FILE: app.py
# Purpose: Streamlit Dashboard (Live Analytics)
# ==========================================================

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Parcl Buyer Segmentation Dashboard",
    layout="wide"
)

st.title("Real Estate Buyer Segmentation & Investment Profiling")
st.markdown("AI-driven buyer intelligence dashboard for Parcl Co. Limited")

# ----------------------------------------------------------
# Load data (no caching, so it always reflects the latest CSV)
# ----------------------------------------------------------
def load_data():
    df = pd.read_csv("final_segmented_data.csv")
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load final_segmented_data.csv. Run the pipeline script first. Error: {e}")
    st.stop()

st.sidebar.markdown(f"**Total Records in File:** {len(df)}")

# ----------------------------------------------------------
# Sidebar Filters
# ----------------------------------------------------------
st.sidebar.header("Filters")

country_options = ["All"] + sorted(df['country'].dropna().unique().tolist())
region_options = ["All"] + sorted(df['region'].dropna().unique().tolist())
purpose_options = ["All"] + sorted(df['acquisition_purpose'].dropna().unique().tolist())
client_type_options = ["All"] + sorted(df['client_type'].dropna().unique().tolist())

selected_country = st.sidebar.selectbox("Country", country_options, key="country_filter")
selected_region = st.sidebar.selectbox("Region", region_options, key="region_filter")
selected_purpose = st.sidebar.selectbox("Acquisition Purpose", purpose_options, key="purpose_filter")
selected_client_type = st.sidebar.selectbox("Client Type", client_type_options, key="clienttype_filter")

filtered_df = df.copy()

if selected_country != "All":
    filtered_df = filtered_df[filtered_df['country'] == selected_country]
if selected_region != "All":
    filtered_df = filtered_df[filtered_df['region'] == selected_region]
if selected_purpose != "All":
    filtered_df = filtered_df[filtered_df['acquisition_purpose'] == selected_purpose]
if selected_client_type != "All":
    filtered_df = filtered_df[filtered_df['client_type'] == selected_client_type]

st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df)}")

# ----------------------------------------------------------
# Top-level KPIs
# ----------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Clients", filtered_df['client_id'].nunique())
col2.metric("Total Transactions", len(filtered_df))

if 'satisfaction_score' in filtered_df.columns and len(filtered_df) > 0:
    col3.metric("Avg Satisfaction", round(filtered_df['satisfaction_score'].mean(), 2))
else:
    col3.metric("Avg Satisfaction", "N/A")

if 'sale_price' in filtered_df.columns and len(filtered_df) > 0:
    col4.metric("Avg Sale Price", f"{filtered_df['sale_price'].mean():,.0f}")
else:
    col4.metric("Avg Sale Price", "N/A")

st.markdown("---")

# ----------------------------------------------------------
# Stop early with a clear message if filters return nothing
# ----------------------------------------------------------
if len(filtered_df) == 0:
    st.warning("No records match the selected filters. Try changing or resetting the filters in the sidebar.")
    st.stop()

# ----------------------------------------------------------
# MODULE 1: Buyer Segmentation Overview
# ----------------------------------------------------------
st.header("1. Buyer Segmentation Overview")

if 'Segment_Name' in filtered_df.columns:
    seg_counts = filtered_df['Segment_Name'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Count']

    c1, c2 = st.columns(2)
    with c1:
        fig_bar = px.bar(seg_counts, x='Segment', y='Count', color='Segment',
                          title="Client Count per Segment")
        st.plotly_chart(fig_bar, use_container_width=True)
    with c2:
        fig_pie = px.pie(seg_counts, names='Segment', values='Count',
                          title="Segment Share (%)")
        st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.warning("Segment_Name column not found. Run buyer_segmentation_pipeline.py first.")

st.markdown("---")

# ----------------------------------------------------------
# MODULE 2: Investor Behavior Dashboard
# ----------------------------------------------------------
st.header("2. Investor Behavior Dashboard")

c1, c2 = st.columns(2)
with c1:
    if 'sale_price' in filtered_df.columns and 'Segment_Name' in filtered_df.columns:
        fig_box = px.box(filtered_df, x='Segment_Name', y='sale_price', color='Segment_Name',
                          title="Sale Price Distribution by Segment")
        st.plotly_chart(fig_box, use_container_width=True)
with c2:
    if 'loan_applied' in filtered_df.columns and 'Segment_Name' in filtered_df.columns:
        loan_counts = filtered_df.groupby(['Segment_Name', 'loan_applied']).size().reset_index(name='Count')
        fig_loan = px.bar(loan_counts, x='Segment_Name', y='Count', color='loan_applied',
                           barmode='group', title="Loan Applied Status by Segment")
        st.plotly_chart(fig_loan, use_container_width=True)

if 'acquisition_purpose' in filtered_df.columns and 'Segment_Name' in filtered_df.columns:
    purpose_seg = filtered_df.groupby(['Segment_Name', 'acquisition_purpose']).size().reset_index(name='Count')
    fig_purpose = px.bar(purpose_seg, x='Segment_Name', y='Count', color='acquisition_purpose',
                          barmode='stack', title="Acquisition Purpose Mix by Segment")
    st.plotly_chart(fig_purpose, use_container_width=True)

# ----------------------------------------------------------
# MODULE 2 EXTENSION: Average Age & Satisfaction Score per Cluster
# ----------------------------------------------------------
st.subheader("Average Age & Satisfaction Score by Segment")

if 'Segment_Name' in filtered_df.columns:
    age_satisfaction = filtered_df.groupby('Segment_Name').agg(
        Avg_Age=('Age', 'mean'),
        Avg_Satisfaction=('satisfaction_score', 'mean')
    ).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        fig_age = px.bar(age_satisfaction, x='Segment_Name', y='Avg_Age', color='Segment_Name',
                          title="Average Age per Segment")
        st.plotly_chart(fig_age, use_container_width=True)
    with c2:
        fig_sat = px.bar(age_satisfaction, x='Segment_Name', y='Avg_Satisfaction', color='Segment_Name',
                          title="Average Satisfaction Score per Segment")
        st.plotly_chart(fig_sat, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------
# MODULE 3: Geographic Buyer Analysis
# ----------------------------------------------------------
st.header("3. Geographic Buyer Analysis")

if 'region' in filtered_df.columns:
    region_agg_dict = {'Client_Count': ('client_id', 'nunique')}
    if 'sale_price' in filtered_df.columns:
        region_agg_dict['Avg_Sale_Price'] = ('sale_price', 'mean')

    region_data = filtered_df.groupby('region').agg(**region_agg_dict).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        fig_region_count = px.bar(region_data, x='region', y='Client_Count', color='region',
                                   title="Buyer Count by Region")
        st.plotly_chart(fig_region_count, use_container_width=True)
    with c2:
        if 'country' in filtered_df.columns:
            country_data = filtered_df['country'].value_counts().reset_index()
            country_data.columns = ['Country', 'Count']
            fig_country = px.bar(country_data, x='Country', y='Count', color='Country',
                                  title="Buyer Count by Country")
            st.plotly_chart(fig_country, use_container_width=True)

    if 'Segment_Name' in filtered_df.columns:
        region_segment = filtered_df.groupby(['region', 'Segment_Name']).size().reset_index(name='Count')
        fig_region_seg = px.bar(region_segment, x='region', y='Count', color='Segment_Name',
                                 barmode='stack', title="Segment Distribution Across Regions")
        st.plotly_chart(fig_region_seg, use_container_width=True)
else:
    st.warning("Region column not found in data.")

st.markdown("---")

# ----------------------------------------------------------
# Segment Summary Cards (per segment quick stats)
# ----------------------------------------------------------
st.header("Segment Summary Cards")

if 'Segment_Name' in filtered_df.columns:
    segments_list = filtered_df['Segment_Name'].unique()
    cards_per_row = 2
    rows_needed = (len(segments_list) + cards_per_row - 1) // cards_per_row

    seg_idx = 0
    for r in range(rows_needed):
        cols = st.columns(cards_per_row)
        for c in cols:
            if seg_idx >= len(segments_list):
                break
            seg_name = segments_list[seg_idx]
            seg_data = filtered_df[filtered_df['Segment_Name'] == seg_name]
            with c:
                st.markdown(f"### {seg_name}")
                st.metric("Clients", seg_data['client_id'].nunique())
                st.metric("Avg Age", round(seg_data['Age'].mean(), 1) if len(seg_data) > 0 else "N/A")
                st.metric("Avg Satisfaction", round(seg_data['satisfaction_score'].mean(), 2) if len(seg_data) > 0 else "N/A")
            seg_idx += 1

st.markdown("---")

# ----------------------------------------------------------
# MODULE 4: Segment Insights Panel
# ----------------------------------------------------------
st.header("4. Segment Insights Panel")

if 'Segment_Name' in filtered_df.columns:
    insights_agg_dict = {
        'Total_Clients': ('client_id', 'nunique'),
        'Avg_Age': ('Age', 'mean'),
        'Avg_Satisfaction': ('satisfaction_score', 'mean'),
        'Loan_Applied_Rate': ('loan_applied', lambda x: (x.astype(str).str.lower() == 'yes').mean())
    }
    if 'sale_price' in filtered_df.columns:
        insights_agg_dict['Avg_Sale_Price'] = ('sale_price', 'mean')

    insights = filtered_df.groupby('Segment_Name').agg(**insights_agg_dict).reset_index()

    insights['Avg_Age'] = insights['Avg_Age'].round(1)
    insights['Avg_Satisfaction'] = insights['Avg_Satisfaction'].round(2)
    if 'Avg_Sale_Price' in insights.columns:
        insights['Avg_Sale_Price'] = insights['Avg_Sale_Price'].apply(lambda x: f"{x:,.0f}")
    insights['Loan_Applied_Rate'] = (insights['Loan_Applied_Rate'] * 100).round(1).astype(str) + '%'

    st.dataframe(insights, use_container_width=True)

    selected_segment = st.selectbox("View detailed records for segment:", insights['Segment_Name'].unique())
    st.dataframe(filtered_df[filtered_df['Segment_Name'] == selected_segment], use_container_width=True)
else:
    st.warning("Segment_Name column not found.")

st.markdown("---")

# ----------------------------------------------------------
# Recommended Buyer Segments (Dynamic - reflects current filters)
# ----------------------------------------------------------
st.header("Recommended Buyer Segments")

segment_cluster_map = {
    "Global Investors": ("C1", "High income, investment purchases"),
    "First-Time Buyers": ("C2", "Younger, loan dependent"),
    "Corporate Buyers": ("C3", "Companies purchasing multiple units"),
    "Luxury Investors": ("C4", "High satisfaction, large investments")
}

if 'Segment_Name' in filtered_df.columns:
    rows = []
    for segment_name, (cluster_code, characteristics) in segment_cluster_map.items():
        segment_data = filtered_df[filtered_df['Segment_Name'] == segment_name]
        client_count = segment_data['client_id'].nunique()
        avg_age = f"{segment_data['Age'].mean():.1f}" if len(segment_data) > 0 else "N/A"
        avg_satisfaction = f"{segment_data['satisfaction_score'].mean():.2f}" if len(segment_data) > 0 else "N/A"

        if 'sale_price' in segment_data.columns and len(segment_data) > 0:
            avg_price = f"{segment_data['sale_price'].mean():,.0f}"
        else:
            avg_price = "N/A"

        rows.append({
            "Cluster": cluster_code,
            "Buyer Type": segment_name,
            "Characteristics": characteristics,
            "Client Count": client_count,
            "Avg Age": avg_age,
            "Avg Satisfaction": avg_satisfaction,
            "Avg Sale Price": avg_price
        })

    recommended_df = pd.DataFrame(rows)
    st.table(recommended_df)
else:
    st.warning("Segment_Name column not found. Run buyer_segmentation_pipeline.py first.")

st.markdown("---")

# ----------------------------------------------------------
# Cluster Evaluation Metrics
# ----------------------------------------------------------
st.header("Cluster Evaluation Metrics")

try:
    eval_metrics = pd.read_csv("cluster_evaluation_metrics.csv")
    st.table(eval_metrics)
except FileNotFoundError:
    st.warning("cluster_evaluation_metrics.csv not found. Run the pipeline script first.")

st.markdown("---")

# ----------------------------------------------------------
# PCA Cluster Visualizations
# ----------------------------------------------------------
st.header("Cluster Visualizations")

c1, c2 = st.columns(2)
with c1:
    try:
        st.image("pca_kmeans_clusters.png", caption="K-Means Clusters (PCA 2D Projection)", use_container_width=True)
    except Exception:
        st.warning("pca_kmeans_clusters.png not found.")
with c2:
    try:
        st.image("pca_hierarchical_clusters.png", caption="Hierarchical Clusters (PCA 2D Projection)", use_container_width=True)
    except Exception:
        st.warning("pca_hierarchical_clusters.png not found.")

try:
    st.image("hierarchical_dendrogram.png", caption="Hierarchical Clustering Dendrogram", use_container_width=True)
except Exception:
    st.warning("hierarchical_dendrogram.png not found.")

st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Dashboard Built by Satyam Kumar Singh | Parcl Co. Limited | Unified Mentor Project</p>",
    unsafe_allow_html=True
)