from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment_data"

# Use deployment artifacts when available.
# Otherwise use the locally generated processed datasets.
DATA_DIR = (
    DEPLOYMENT_DIR
    if DEPLOYMENT_DIR.exists()
    else PROCESSED_DIR
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FORESIGHT",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    risk_df = pd.read_parquet(
    DATA_DIR / "risk_scoring.parquet"
    )

    predictions_df = pd.read_parquet(
        DATA_DIR / "validation_predictions.parquet"
    )

    predictions_df["week_start"] = pd.to_datetime(
        predictions_df["week_start"]
    )

    return risk_df, predictions_df


risk, predictions = load_data()


# ============================================================
# HEADER
# ============================================================

st.title("📦 FORESIGHT")

st.subheader(
    "Inventory Intelligence & Demand Forecasting"
)

latest_week = predictions["week_start"].max()

st.caption(
    f"Data / forecast period: {latest_week.strftime('%d %b %Y')} "
    f"• Forecast horizon: 4 weeks"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("FORESIGHT")

st.sidebar.markdown(
    "### 🔎 Dashboard Filters"
)

categories = sorted(
    risk["category"]
    .dropna()
    .unique()
)

selected_categories = st.sidebar.multiselect(
    "Category",
    categories,
    default=categories,
    help="Filter the dashboard by product category."
)

risk_options = sorted(
    risk["risk"]
    .dropna()
    .unique()
)

selected_risks = st.sidebar.multiselect(
    "Risk",
    risk_options,
    default=risk_options,
    help="Filter the dashboard by inventory-risk category."
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **📘 Need help?**

    Use the **Guide** page in the navigation
    menu for detailed instructions, risk
    definitions, KPI explanations, forecasting
    methodology and data limitations.
    """
)


# ============================================================
# FILTER DATA
# ============================================================

filtered = risk[
    risk["category"].isin(
        selected_categories
    )
    &
    risk["risk"].isin(
        selected_risks
    )
].copy()


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

st.markdown("## Executive Overview")

total_skus = len(filtered)

stockout_count = (
    filtered["risk"]
    .eq("STOCKOUT RISK")
    .sum()
)

watch_count = (
    filtered["risk"]
    .eq("WATCH")
    .sum()
)

overstock_count = (
    filtered["risk"]
    .eq("OVERSTOCK")
    .sum()
)

healthy_count = (
    filtered["risk"]
    .eq("HEALTHY")
    .sum()
)

inventory_available = (
    filtered["inventory_data_available"]
    .sum()
)

inventory_coverage = (
    inventory_available / total_skus * 100
    if total_skus > 0
    else 0
)

stockout_exposure = (
    filtered["stockout_exposure"]
    .sum()
)

excess_value = (
    filtered["excess_inventory_value"]
    .sum()
)


# ============================================================
# KPI TABLE
# ============================================================

kpi_data = pd.DataFrame({
    "Metric": [
        "Total SKUs",
        "🔴 Stockout Risk",
        "🟡 Watch",
        "🔵 Overstock",
        "🟢 Healthy",
        "💰 Stockout Exposure",
        "📦 Inventory Coverage",
    ],

    "Value": [
        f"{total_skus:,}",
        f"{stockout_count:,}",
        f"{watch_count:,}",
        f"{overstock_count:,}",
        f"{healthy_count:,}",
        f"₹{stockout_exposure:,.0f}",
        f"{inventory_coverage:.1f}%",
    ],

    "Description": [
        "Products monitored",
        "Immediate attention",
        "Review required",
        "Excess inventory",
        "Within acceptable range",
        "Estimated stockout exposure",
        "SKUs with inventory data",
    ]
})

st.dataframe(
    kpi_data,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Metric": st.column_config.TextColumn(
            "Metric",
            width="medium"
        ),
        "Value": st.column_config.TextColumn(
            "Value",
            width="medium"
        ),
        "Description": st.column_config.TextColumn(
            "What it means",
            width="large"
        ),
    }
)


# ============================================================
# RISK + FINANCIAL EXPOSURE
# ============================================================

st.markdown("---")

left, right = st.columns(2)


with left:

    st.subheader(
        "Inventory Risk Distribution"
    )

    risk_counts = (
        filtered["risk"]
        .value_counts()
        .reset_index()
    )

    risk_counts.columns = [
        "risk",
        "count"
    ]

    fig = px.bar(
        risk_counts,
        x="risk",
        y="count",
        text="count",
        title="SKUs by Risk Category",
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="SKUs",
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with right:

    st.subheader(
        "Financial Exposure"
    )

    exposure_data = pd.DataFrame({
        "Metric": [
            "Stockout Exposure",
            "Excess Inventory"
        ],

        "Value": [
            stockout_exposure,
            excess_value
        ]
    })

    fig = px.bar(
        exposure_data,
        x="Metric",
        y="Value",
        text="Value",
        title="Estimated Financial Impact",
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="₹",
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# DECISIONING GRID
# ============================================================

st.markdown("---")

st.subheader(
    "Inventory Decisioning Grid"
)

st.caption(
    "Higher stockout pressure indicates potential demand "
    "shortage; higher overstock pressure indicates excess "
    "inventory relative to the 4-week forecast."
)

grid_data = filtered.copy()

grid_data["stockout_pressure"] = (
    grid_data["forecast_horizon_demand"]
    - grid_data["inventory_position"]
)

grid_data["overstock_pressure"] = (
    grid_data["inventory_position"]
    - grid_data["forecast_horizon_demand"]
)

fig = px.scatter(
    grid_data,
    x="stockout_pressure",
    y="overstock_pressure",
    color="risk",
    hover_data=[
        "sku_id",
        "sku_name",
        "category",
        "forecast_weekly_demand",
        "inventory_position",
        "weeks_of_cover",
        "recommended_action",
    ],
    title="Stockout vs Overstock Pressure",
    labels={
        "stockout_pressure": "Stockout Pressure",
        "overstock_pressure": "Overstock Pressure",
        "risk": "Risk",
    },
)

fig.add_hline(
    y=0,
    line_dash="dash"
)

fig.add_vline(
    x=0,
    line_dash="dash"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# STOCKOUT RISKS
# ============================================================

st.markdown("---")

st.subheader(
    "🔴 Top Stockout Risks"
)

st.caption(
    "Prioritized by estimated stockout exposure."
)

stockout_table = (
    filtered[
        filtered["risk"]
        == "STOCKOUT RISK"
    ]
    .sort_values(
        "stockout_exposure",
        ascending=False
    )
    .head(20)
)

st.dataframe(
    stockout_table[
        [
            "sku_id",
            "sku_name",
            "category",
            "forecast_weekly_demand",
            "inventory_position",
            "weeks_of_cover",
            "stockout_exposure",
            "recommended_action",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# OVERSTOCK
# ============================================================

st.markdown("---")

st.subheader(
    "🔵 Top Overstock / Slow-Moving SKUs"
)

st.caption(
    "Prioritized by estimated excess inventory value."
)

overstock_table = (
    filtered[
        filtered["risk"]
        == "OVERSTOCK"
    ]
    .sort_values(
        "excess_inventory_value",
        ascending=False
    )
    .head(20)
)

st.dataframe(
    overstock_table[
        [
            "sku_id",
            "sku_name",
            "category",
            "forecast_weekly_demand",
            "inventory_position",
            "weeks_of_cover",
            "excess_inventory_value",
            "recommended_action",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# SKU ANALYSIS
# ============================================================

st.markdown("---")

st.header("SKU Analysis")

sku_list = sorted(
    filtered["sku_id"]
    .dropna()
    .unique()
)

if sku_list:

    selected_sku = st.selectbox(
        "Select SKU",
        sku_list,
        help="Select an individual SKU for detailed analysis."
    )

    sku_row = filtered[
        filtered["sku_id"]
        == selected_sku
    ].iloc[0]

    # --------------------------------------------------------
    # SKU DETAILS
    # --------------------------------------------------------

    st.subheader("SKU Details")

    sku_details = pd.DataFrame({
        "Field": [
            "SKU",
            "Product",
            "Category",
            "Subcategory",
            "Brand",
            "Risk",
            "Recommended Action",
        ],

        "Value": [
            sku_row["sku_id"],
            sku_row["sku_name"],
            sku_row["category"],
            sku_row["subcategory"],
            sku_row["brand"],
            sku_row["risk"],
            sku_row["recommended_action"],
        ]
    })

    st.dataframe(
        sku_details,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # INVENTORY METRICS
    # --------------------------------------------------------

    st.subheader("Demand & Inventory Position")

    inventory_details = pd.DataFrame({
        "Metric": [
            "Forecast Demand / Week",
            "4-Week Forecast Demand",
            "Inventory Position",
            "Weeks of Cover",
            "Reorder Point",
            "Safety Stock",
        ],

        "Value": [
            f"{sku_row['forecast_weekly_demand']:,.2f}",
            f"{sku_row['forecast_horizon_demand']:,.2f}",
            f"{sku_row['inventory_position']:,.0f}",
            f"{sku_row['weeks_of_cover']:,.2f}",
            f"{sku_row['reorder_point']:,.0f}",
            f"{sku_row['safety_stock']:,.0f}",
        ]
    })

    st.dataframe(
        inventory_details,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # FINANCIAL IMPACT
    # --------------------------------------------------------

    st.subheader("Financial Impact")

    financial_details = pd.DataFrame({
        "Metric": [
            "Stockout Exposure",
            "Excess Inventory Value",
        ],

        "Value": [
            f"₹{sku_row['stockout_exposure']:,.2f}",
            f"₹{sku_row['excess_inventory_value']:,.2f}",
        ]
    })

    st.dataframe(
        financial_details,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FORECAST VS ACTUAL
# ============================================================

st.markdown("---")

st.header("Forecast vs Actual Demand")

selected_forecast_sku = st.selectbox(
    "Select SKU for forecast comparison",
    sorted(
        risk["sku_id"]
        .dropna()
        .unique()
    ),
    key="forecast_sku",
    help="Compare historical demand with the model forecast."
)

sku_forecast = predictions[
    predictions["sku_id"]
    == selected_forecast_sku
].copy()

if sku_forecast.empty:

    st.info(
        "No validation forecast history is available "
        "for this SKU."
    )

else:

    sku_forecast = sku_forecast.sort_values(
        "week_start"
    )

    fig = px.line(
        sku_forecast,
        x="week_start",
        y=[
            "units_sold",
            "prediction"
        ],
        markers=True,
        labels={
            "units_sold": "Actual Demand",
            "prediction": "LightGBM Forecast",
            "week_start": "Week",
            "value": "Units",
        },
        title=(
            f"Actual vs LightGBM Forecast — "
            f"{selected_forecast_sku}"
        )
    )

    fig.update_layout(
        legend_title_text=""
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "FORESIGHT • Demand Forecasting & Inventory Risk Intelligence "
    "• Prototype"
)