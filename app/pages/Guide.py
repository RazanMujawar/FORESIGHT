import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FORESIGHT — Guide",
    page_icon="📘",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("📘 FORESIGHT User Guide")

st.markdown(
    """
    Welcome to **FORESIGHT — Inventory Intelligence & Demand Forecasting**.

    FORESIGHT combines historical sales, product information,
    inventory data and machine-learning demand forecasts to help
    identify inventory risks and prioritize business actions.
    """
)


# ============================================================
# 1. GETTING STARTED
# ============================================================

st.header("1. Getting Started")

st.markdown(
    """
    ### Step 1 — Apply filters

    Use the filters in the Dashboard sidebar to focus your analysis
    on specific product categories or risk categories.

    ### Step 2 — Review the Executive Overview

    Start with the KPI section to understand the overall inventory
    situation.

    ### Step 3 — Investigate risks

    Review the Stockout Risk and Overstock sections to identify
    products requiring attention.

    ### Step 4 — Analyze individual SKUs

    Use the SKU Analysis section to inspect a specific product,
    including its forecast, inventory position, coverage and
    recommended action.

    ### Step 5 — Review forecast performance

    Use the Forecast vs Actual section to compare historical demand
    with the LightGBM forecast.
    """
)


# ============================================================
# 2. DASHBOARD SECTIONS
# ============================================================

st.header("2. Understanding the Dashboard")

sections = {
    "Executive Overview":
        "Provides a high-level snapshot of the number of monitored SKUs, "
        "inventory risks, financial exposure and inventory data coverage.",

    "Risk Distribution":
        "Shows how monitored SKUs are distributed across the different "
        "inventory risk categories.",

    "Financial Exposure":
        "Shows estimated stockout exposure and excess inventory value "
        "associated with the current inventory position.",

    "Inventory Decisioning Grid":
        "Visualizes stockout pressure against overstock pressure to "
        "help prioritize inventory decisions.",

    "Top Stockout Risks":
        "Lists high-priority SKUs where current inventory may not be "
        "sufficient for expected near-term demand.",

    "Top Overstock / Slow Movers":
        "Highlights products with substantial inventory relative to "
        "forecast demand.",

    "SKU Analysis":
        "Provides detailed information for an individual SKU, including "
        "forecast demand, inventory, coverage, risk and recommended action.",

    "Forecast vs Actual":
        "Compares historical observed demand with the LightGBM forecast "
        "for a selected SKU."
}

for title, description in sections.items():

    with st.expander(title):

        st.write(description)


# ============================================================
# 3. RISK DEFINITIONS
# ============================================================

st.header("3. Risk Categories")

risk_definitions = {
    "🔴 STOCKOUT RISK":
        "Inventory may be insufficient to support expected demand "
        "during the assumed replenishment lead-time period. "
        "Recommended action: expedite replenishment.",

    "🟡 WATCH":
        "Inventory requires review because it is approaching a "
        "replenishment threshold. Recommended action: review the "
        "replenishment plan.",

    "🟢 HEALTHY":
        "Inventory is currently within an acceptable range relative "
        "to forecast demand. Recommended action: maintain current "
        "inventory.",

    "🔵 OVERSTOCK":
        "Inventory is high relative to expected demand, indicating "
        "potential excess inventory. Recommended action: reduce or "
        "defer replenishment.",

    "⚪ NO INVENTORY DATA":
        "Current inventory information is unavailable for the SKU. "
        "The system cannot make a reliable inventory-risk decision."
}

for risk_name, description in risk_definitions.items():

    st.subheader(risk_name)
    st.write(description)


# ============================================================
# 4. KPI DEFINITIONS
# ============================================================

st.header("4. KPI Definitions")

kpis = {
    "Total SKUs":
        "Number of SKUs currently included in the risk analysis.",

    "Stockout Risk":
        "Number of SKUs currently classified as having potential "
        "near-term stockout risk.",

    "Watch":
        "Number of SKUs requiring inventory review.",

    "Overstock":
        "Number of SKUs with inventory substantially above expected "
        "demand.",

    "Stockout Exposure":
        "Estimated cost-value exposure from inventory shortfalls over "
        "the forecast horizon.",

    "Inventory Data Coverage":
        "Percentage of monitored SKUs for which current inventory "
        "information is available.",

    "Weeks of Cover":
        "Estimated number of weeks that the current inventory position "
        "can support the forecast weekly demand."
}

for name, definition in kpis.items():

    st.markdown(
        f"**{name}**  \n{definition}"
    )


# ============================================================
# 5. FORECASTING
# ============================================================

st.header("5. Demand Forecasting")

st.markdown(
    """
    FORESIGHT uses a **LightGBM regression model** to forecast
    weekly SKU-level demand.

    The model uses historical demand patterns including:

    - Recent demand lags
    - Short-term rolling demand
    - Seasonal lags
    - Week of year
    - Month
    - Quarter
    - SKU attributes
    - Product category and brand information
    """
)

st.subheader("Model Performance")

st.markdown(
    """
    **Rolling-origin backtest**

    | Model | Average WAPE |
    |---|---:|
    | Seasonal Naive | **37.09%** |
    | LightGBM | **27.81%** |

    **Relative improvement: 25.02%**
    """
)


# ============================================================
# 6. INVENTORY RISK METHODOLOGY
# ============================================================

st.header("6. Inventory Risk Methodology")

st.markdown(
    """
    FORESIGHT combines forecast demand with the available inventory
    position to estimate inventory risk.

    ### Inventory Position

    **Inventory Position = On Hand + On Order**

    The supplied dataset does not provide a populated on-order
    inventory field, so the current implementation treats on-order
    inventory as zero.

    ### Forecast Horizon

    The risk engine evaluates expected demand over a **4-week
    forecast horizon**.

    ### Lead Time

    The source data does not provide an observed lead-time field.
    The current implementation therefore uses an explicit modelling
    assumption of **1 week** for the replenishment lead-time calculation.
    """
)


# ============================================================
# 7. DATA QUALITY & LIMITATIONS
# ============================================================

st.header("7. Data Quality & Limitations")

st.markdown(
    """
    ### Inventory coverage

    The source inventory snapshot does not contain inventory records
    for all SKUs.

    **505 of 5,000 SKUs do not have inventory records.**

    These products are therefore classified as:

    **NO INVENTORY DATA**

    rather than being assigned an artificial inventory-risk level.

    ### Forecast interpretation

    Forecast accuracy is measured using WAPE and evaluated using
    rolling-origin backtesting to preserve the time-series nature
    of the problem.

    ### Financial impact

    Financial exposure values are estimates based on the available
    forecast, inventory and cost information. They should be treated
    as decision-support estimates rather than accounting figures.
    """
)


# ============================================================
# 8. RECOMMENDED WORKFLOW
# ============================================================

st.header("8. Recommended Analyst Workflow")

st.markdown(
    """
    A practical workflow for using FORESIGHT is:

    **1. Start with the Executive Overview**

    Identify whether stockout or excess-inventory exposure is the
    dominant concern.

    **2. Filter the relevant category**

    Narrow the analysis to the product area requiring attention.

    **3. Review Stockout Risk**

    Prioritize products by financial exposure and expected demand.

    **4. Review Overstock**

    Identify slow-moving products with substantial inventory.

    **5. Inspect individual SKUs**

    Use SKU Analysis to understand the forecast, inventory coverage
    and recommended action.

    **6. Use Forecast vs Actual**

    Review forecast behavior and historical demand patterns.

    **7. Take action**

    Use the recommended action as a starting point for replenishment,
    inventory reduction or further review.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "FORESIGHT — Inventory Intelligence & Demand Forecasting"
)