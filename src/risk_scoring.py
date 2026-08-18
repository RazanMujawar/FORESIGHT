from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

FORECAST_HORIZON_WEEKS = 4

# The source dataset does not provide lead time.
# This is an explicit modelling assumption.
ASSUMED_LEAD_TIME_WEEKS = 1


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    predictions = pd.read_parquet(
        PROCESSED_DIR
        / "validation_predictions.parquet"
    )

    inventory = pd.read_parquet(
        PROCESSED_DIR
        / "inventory_latest.parquet"
    )

    sku = pd.read_parquet(
        PROCESSED_DIR
        / "sku_master.parquet"
    )

    predictions["week_start"] = pd.to_datetime(
        predictions["week_start"]
    )

    return predictions, inventory, sku


# ============================================================
# BUILD RISK DATASET
# ============================================================

def build_risk_dataset(
    predictions,
    inventory,
    sku
):

    # --------------------------------------------------------
    # Use latest available model predictions
    # --------------------------------------------------------

    latest_week = predictions[
        "week_start"
    ].max()

    latest_predictions = predictions[
        predictions["week_start"]
        == latest_week
    ].copy()

    latest_predictions = latest_predictions[
        [
            "sku_id",
            "week_start",
            "units_sold",
            "prediction",
        ]
    ]

    latest_predictions = latest_predictions.rename(
        columns={
            "prediction":
            "forecast_weekly_demand"
        }
    )

    # --------------------------------------------------------
    # Inventory
    # --------------------------------------------------------

    risk = latest_predictions.merge(
        inventory,
        on="sku_id",
        how="left"
    )

    # --------------------------------------------------------
    # SKU attributes
    # --------------------------------------------------------

    risk = risk.merge(
        sku[
            [
                "sku_id",
                "sku_name",
                "category",
                "subcategory",
                "brand",
                "cost_price",
            ]
        ],
        on="sku_id",
        how="left"
    )

    # --------------------------------------------------------
    # Inventory availability
    # --------------------------------------------------------

    risk["inventory_data_available"] = (
        risk["on_hand_units"].notna()
    ).astype(int)

    # --------------------------------------------------------
    # On-order assumption
    # --------------------------------------------------------

    risk["on_order_units"] = (
        risk["on_order_units"]
        .fillna(0)
    )

    risk["inventory_position"] = (
        risk["on_hand_units"].fillna(0)
        + risk["on_order_units"]
    )

    # --------------------------------------------------------
    # Forecast horizon
    # --------------------------------------------------------

    risk["forecast_horizon_demand"] = (
        risk["forecast_weekly_demand"]
        * FORECAST_HORIZON_WEEKS
    )

    # --------------------------------------------------------
    # Weeks of cover
    # --------------------------------------------------------

    risk["weeks_of_cover"] = np.where(
        risk["forecast_weekly_demand"] > 0,
        risk["inventory_position"]
        / risk["forecast_weekly_demand"],
        np.nan
    )

    # --------------------------------------------------------
    # Lead-time demand
    # --------------------------------------------------------

    risk["lead_time_demand"] = (
        risk["forecast_weekly_demand"]
        * ASSUMED_LEAD_TIME_WEEKS
    )

    # --------------------------------------------------------
    # Expected inventory after forecast horizon
    # --------------------------------------------------------

    risk["projected_inventory_4w"] = (
        risk["inventory_position"]
        - risk["forecast_horizon_demand"]
    )

    return risk


# ============================================================
# CLASSIFY RISK
# ============================================================

def classify_risk(row):

    # No inventory data
    if row["inventory_data_available"] == 0:

        return "NO INVENTORY DATA"

    forecast = row["forecast_weekly_demand"]
    inventory_position = row["inventory_position"]
    reorder_point = row["reorder_point"]

    if forecast <= 0:

        if inventory_position > reorder_point * 2:
            return "OVERSTOCK"

        return "HEALTHY"

    # Stockout risk within assumed lead time
    if inventory_position < row["lead_time_demand"]:

        return "STOCKOUT RISK"

    # Below reorder point
    if inventory_position < reorder_point:

        return "WATCH"

    # Excessive coverage
    if row["weeks_of_cover"] > 12:

        return "OVERSTOCK"

    return "HEALTHY"


# ============================================================
# RECOMMENDED ACTION
# ============================================================

def recommended_action(risk):

    actions = {

        "STOCKOUT RISK":
            "Expedite replenishment",

        "WATCH":
            "Review replenishment plan",

        "HEALTHY":
            "Maintain current inventory",

        "OVERSTOCK":
            "Reduce / defer replenishment",

        "NO INVENTORY DATA":
            "Obtain current inventory position",
    }

    return actions.get(
        risk,
        "Review"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FORESIGHT — INVENTORY RISK SCORING")
    print("=" * 70)

    predictions, inventory, sku = (
        load_data()
    )

    print(
        f"\nLatest forecast week: "
        f"{predictions['week_start'].max().date()}"
    )

    print(
        f"Forecast records: "
        f"{len(predictions):,}"
    )

    risk = build_risk_dataset(
        predictions,
        inventory,
        sku
    )

    # --------------------------------------------------------
    # Risk classification
    # --------------------------------------------------------

    risk["risk"] = (
        risk.apply(
            classify_risk,
            axis=1
        )
    )

    risk["recommended_action"] = (
        risk["risk"]
        .apply(recommended_action)
    )

    # --------------------------------------------------------
    # Financial exposure
    # --------------------------------------------------------

    risk["stockout_exposure"] = np.where(
        risk["risk"] == "STOCKOUT RISK",

        np.maximum(
            risk["forecast_horizon_demand"]
            - risk["inventory_position"],
            0
        )
        * risk["cost_price"],

        0
    )

    risk["excess_inventory_value"] = np.where(
        risk["risk"] == "OVERSTOCK",

        np.maximum(
            risk["inventory_position"]
            - risk["forecast_horizon_demand"],
            0
        )
        * risk["cost_price"],

        0
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    risk.to_parquet(
        PROCESSED_DIR
        / "risk_scoring.parquet",
        index=False
    )

    risk.to_csv(
        REPORTS_DIR
        / "risk_scoring.csv",
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RISK SUMMARY")
    print("=" * 70)

    print(
        risk["risk"]
        .value_counts()
        .to_string()
    )

    print("\nFinancial exposure:")

    print(
        f"Stockout exposure: "
        f"₹{risk['stockout_exposure'].sum():,.2f}"
    )

    print(
        f"Excess inventory value: "
        f"₹{risk['excess_inventory_value'].sum():,.2f}"
    )

    print(
        "\n✓ risk_scoring.parquet saved"
    )

    print(
        "✓ risk_scoring.csv saved"
    )


if __name__ == "__main__":
    main()