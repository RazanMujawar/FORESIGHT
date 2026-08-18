from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FORESIGHT",
    page_icon="📦",
    layout="wide",
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    risk_df = pd.read_parquet(
        PROCESSED_DIR
        / "risk_scoring.parquet"
    )

    predictions_df = pd.read_parquet(
        PROCESSED_DIR
        / "validation_predictions.parquet"
    )

    predictions_df["week_start"] = pd.to_datetime(
        predictions_df["week_start"]
    )

    return risk_df, predictions_df


risk, predictions = load_data()



# ============================================================
# HEADER
# ============================================================

st.title("FORESIGHT")
st.subheader(
    "Inventory Intelligence & Demand Forecasting"
)

st.caption(
    "AI-assisted demand forecasting and inventory "
    "risk monitoring"
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Filters")

categories = sorted(
    risk["category"]
    .dropna()
    .unique()
)

selected_categories = st.sidebar.multiselect(
    "Category",
    categories,
    default=categories
)

risk_options = sorted(
    risk["risk"]
    .dropna()
    .unique()
)

selected_risks = st.sidebar.multiselect(
    "Risk",
    risk_options,
    default=risk_options
)


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
# KPI CARDS
# ============================================================

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

stockout_exposure = (
    filtered["stockout_exposure"]
    .sum()
)

excess_value = (
    filtered["excess_inventory_value"]
    .sum()
)


col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "SKUs",
    f"{total_skus:,}"
)

col2.metric(
    "Stockout Risk",
    f"{stockout_count:,}"
)

col3.metric(
    "Watch",
    f"{watch_count:,}"
)

col4.metric(
    "Overstock",
    f"{overstock_count:,}"
)

col5.metric(
    "Stockout Exposure",
    f"₹{stockout_exposure:,.0f}"
)


# ============================================================
# RISK DISTRIBUTION
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
        title="SKUs by Risk Category",
        text="count",
    )

    fig.update_layout(
        xaxis_title="Risk",
        yaxis_title="SKUs",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with right:

    st.subheader(
        "Inventory Value Exposure"
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
        title="Financial Exposure",
        text="Value",
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="₹",
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
        "stockout_pressure":
            "Stockout Pressure",
        "overstock_pressure":
            "Overstock Pressure",
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
    "Top Stockout Risks"
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
    "Top Overstock / Slow-Moving SKUs"
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
# FORECAST VS ACTUAL
# ============================================================

st.markdown("---")

st.subheader(
    "Forecast vs Actual Demand"
)

selected_forecast_sku = st.selectbox(
    "Select SKU for forecast comparison",
    sorted(
        risk["sku_id"]
        .dropna()
        .unique()
    ),
    key="forecast_sku"
)

sku_forecast = predictions[
    predictions["sku_id"]
    == selected_forecast_sku
].copy()

if sku_forecast.empty:

    st.info(
        "No validation forecast history "
        "is available for this SKU."
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
            "value": "Units",
            "week_start": "Week",
            "variable": "Series"
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
# SKU DRILL DOWN
# ============================================================

st.markdown("---")

st.subheader(
    "SKU Drill-Down"
)

sku_list = sorted(
    filtered["sku_id"]
    .dropna()
    .unique()
)

selected_sku = st.selectbox(
    "Select SKU",
    sku_list
)

sku_row = filtered[
    filtered["sku_id"]
    == selected_sku
].iloc[0]


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "SKU",
    sku_row["sku_id"]
)

col2.metric(
    "Forecast / Week",
    f"{sku_row['forecast_weekly_demand']:.1f}"
)

col3.metric(
    "Inventory",
    f"{sku_row['inventory_position']:,.0f}"
)

col4.metric(
    "Weeks of Cover",
    f"{sku_row['weeks_of_cover']:.1f}"
)


st.write(
    f"**Product:** {sku_row['sku_name']}"
)

st.write(
    f"**Category:** {sku_row['category']}"
)

st.write(
    f"**Risk:** {sku_row['risk']}"
)

st.write(
    f"**Recommended Action:** "
    f"{sku_row['recommended_action']}"
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "FORESIGHT — Demand Forecasting & Inventory Risk "
    "Intelligence | Prototype"
)